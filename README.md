# KAIROS Team 2022

Our approach operates in two phases. In phase 1, the algorithm learns from expert demonstrations by integrating the IQ-learn imitation learning algorithm with the VPT model. No human interaction is required during this phase.

In phase 2, **the user is required to give pairwise preferences** through the keyboard based on two videos of different trajectories presented on a GUI. Our code will first collect 100 different trajectories before asking for user input, which might take a few minutes. Current keyboard shortcuts are:  
LEFT ARROW: trajectory displayed on the left side of the GUI is the best performer.  
RIGHT ARROW: trajectory displayed on the right side of the GUI is the best performer.  
UP ARROW: both trajectories are equivalent.  
DOWN ARROW: trajectories are not comparable (for example, performing different behaviors).  
X: stops learning and saves the latest model to disk.  

The main idea is to allocate about 2 hours of human time per task, requiring the human evaluator to press X to end the session.
