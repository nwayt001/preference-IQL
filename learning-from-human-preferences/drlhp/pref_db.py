import os
import collections
import copy
import gzip
import pickle
import queue
import time
import zlib
import logging
from threading import Lock, Thread

import numpy as np
import torch
import cv2

from vpt_agent.openai_vpt.lib.actions import ActionTransformer


ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)


class Segment:
    """
    A short recording of agent's behaviour in the environment,
    consisting of a number of video frames and the rewards it received
    during those frames.
    """

    def __init__(self):
        self.frames = []
        self.rewards = []
        self.actions = []
        self.hash = None

    def append(self, frame, reward, action):
        self.frames.append(frame)
        self.rewards.append(reward)
        self.actions.append(action)

    def finalise(self, seg_id=None):
        if seg_id is not None:
            self.hash = seg_id
        else:
            # This looks expensive, but don't worry -
            # it only takes about 0.5 ms.
            self.hash = hash(np.array(self.frames).tostring())

    def __len__(self):
        return len(self.frames)


class CompressedDict(collections.MutableMapping):

    def __init__(self):
        self.store = dict()

    def __getitem__(self, key):
        return pickle.loads(zlib.decompress(self.store[key]))

    def __setitem__(self, key, value):
        self.store[key] = zlib.compress(pickle.dumps(value))

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key


class PrefDB:
    """
    A circular database of preferences about pairs of segments.

    For each preference, we store the preference itself
    (mu in the paper) and the two segments the preference refers to.
    Segments are stored with deduplication - so that if multiple
    preferences refer to the same segment, the segment is only stored once.
    """

    def __init__(self, maxlen, args):
        self.segments = CompressedDict()
        self.actions = {}
        self.seg_refs = {}
        self.prefs = []
        self.maxlen = maxlen
        self.args = args

        # helper class used to compute action embeddings
        self.num_tiles = args.num_action_tiles
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        # stores data in the format required to compute the rewards
        self.segments_fmt = CompressedDict()
        self.actions_fmt = {}

    def append(self, s1, s2, pref, s1_actions, s2_actions):
        k1 = hash(np.array(s1).tostring())
        k2 = hash(np.array(s2).tostring())

        for k, s, a in zip([k1, k2], [s1, s2], [s1_actions, s2_actions]):
            if k not in self.segments.keys():
                # # might not need to store this raw data
                # self.segments[k] = s
                # self.actions[k] = a

                self.seg_refs[k] = 1

                # compute embedding to store data already in the format expect
                # by the reward head
                self.segments_fmt[k] = self.compute_segments_fmt(s)
                self.actions_fmt[k] = self.compute_actions_fmt(a)

            else:
                self.seg_refs[k] += 1

        tup = (k1, k2, pref)
        self.prefs.append(tup)

        if len(self.prefs) > self.maxlen:
            self.del_first()

    def compute_segments_fmt(self, s):
        '''
        Takes the states of a trajectory in the original format (traj_length, 360, 640, 3) and
        converts to a torch tensor of embeddings (1, traj_length, 128, 128, 3).
        '''
        # create torch tensor to hold data
        traj_length = self.args.preferences.trajectory_length
        s_fmt = torch.zeros(1, traj_length, 128, 128, 3)

        # resize obs images
        for i in range(traj_length):
            s_resized = self.resize_image(s[i], (128, 128))[None]
            s_fmt[:, i, :, :] = torch.from_numpy(s_resized)

        return s_fmt.to(self.args.device)

    def resize_image(self, img, target_resolution):
        # For your sanity, do not resize with any function than INTER_LINEAR
        img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
        return img

    def compute_actions_fmt(self, a):
        '''
        Takes the actions of a trajectory in the original format (list of dicts) and
        converts to a torch tensor of embeddings (1, traj_length, action_embedding_dimension).
        '''
        # create torch tensor to hold data
        traj_length = self.args.preferences.trajectory_length
        # TODO: config value of action_embedding_dimension?
        # size 1034 using the 2x model
        a_fmt = torch.zeros(1, traj_length, self.args.action_emb_size)

        # convert actions to embedding
        for i in range(traj_length):
            # convert dict action to numpy
            policy_repr = self.action_transformer.env2policy(a[i].item())
            action_list = list(policy_repr.values())
            action_np = np.concatenate(action_list, axis=-1)

            # from numpy to embeddings
            env_action_emb = self.embed_function(action_np)

            # add to torch tensor
            a_fmt[:, i, :] = torch.from_numpy(env_action_emb)

        return a_fmt.to(self.args.device)

    def embed_function(self, action):        
        return np.tile(action, self.num_tiles)

    def del_first(self):
        self.del_pref(0)

    def del_pref(self, n):
        if n >= len(self.prefs):
            raise IndexError("Preference {} doesn't exist".format(n))
        k1, k2, _ = self.prefs[n]
        for k in [k1, k2]:
            if self.seg_refs[k] == 1:
                del self.segments[k]
                del self.actions[k]
                del self.seg_refs[k]
            else:
                self.seg_refs[k] -= 1
        del self.prefs[n]

    def __len__(self):
        return len(self.prefs)

    def save(self, path):
        # use tmp file name while saving the file, later rename it
        with open(path+'.tmp', 'wb') as pkl_file:
            pickle.dump(copy.deepcopy(self), pkl_file)
        os.rename(path+'.tmp', path)

    @staticmethod
    def load(path):
        with open(path, 'rb') as pkl_file:
            pref_db = pickle.load(pkl_file)
        return pref_db


class PrefBuffer:
    """
    A helper class to manage asynchronous receiving of preferences on a
    background thread.
    """
    def __init__(self, db_train, db_val):
        self.train_db = db_train
        self.val_db = db_val
        self.lock = Lock()
        self.stop_recv = False

    def start_recv_thread(self, pref_pipe):
        self.stop_recv = False
        Thread(target=self.recv_prefs, args=(pref_pipe, )).start()

    def stop_recv_thread(self):
        self.stop_recv = True

    def recv_prefs(self, pref_pipe):
        n_recvd = 0
        while not self.stop_recv:
            try:
                s1, s2, pref, s1_actions, s2_actions = pref_pipe.get(block=True, timeout=1)
                logging.debug("Pref DB got segment pair with actions plus preferences from pref pipe")
            except queue.Empty:
                logging.debug("Pref DB got no segments")
                continue
            n_recvd += 1

            val_fraction = self.val_db.maxlen / (self.val_db.maxlen +
                                                 self.train_db.maxlen)

            self.lock.acquire(blocking=True)
            if np.random.rand() < val_fraction:
                self.val_db.append(s1, s2, pref, s1_actions, s2_actions)
                # TODO: easy_tf_log not compatible with tf2. replace with regular tf logs
                #easy_tf_log.tflog('val_db_len', len(self.val_db))
            else:
                self.train_db.append(s1, s2, pref, s1_actions, s2_actions)
                # TODO: easy_tf_log not compatible with tf2. replace with regular tf logs
                #easy_tf_log.tflog('train_db_len', len(self.train_db))

            self.lock.release()
            # TODO: easy_tf_log not compatible with tf2. replace with regular tf logs
            #easy_tf_log.tflog('n_prefs_recvd', n_recvd)

    def train_db_len(self):
        return len(self.train_db)

    def val_db_len(self):
        return len(self.val_db)

    def get_dbs(self):
        self.lock.acquire(blocking=True)
        train_copy = copy.deepcopy(self.train_db)
        val_copy = copy.deepcopy(self.val_db)
        self.lock.release()
        return train_copy, val_copy

    def wait_until_len(self, min_len):
        while True:
            self.lock.acquire()
            train_len = len(self.train_db)
            val_len = len(self.val_db)
            self.lock.release()
            if train_len >= min_len and val_len > 0:
                break
            print("Waiting for preferences; {}/{} train so far, {}/{} val ".format(train_len,
                                                                                   min_len,
                                                                                   val_len,
                                                                                   1))
            time.sleep(5.0)
