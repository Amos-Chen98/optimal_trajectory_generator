# optimal trajectory generator
## About

This is a lightweight Python implementation of a trajectory generator. Set some waypoints in RViz through a few clicks, and it will generate a minimum jerk trajectory in milliseconds!

## Usage

### 1. Install

```
git clone https://github.com/Amos-Chen98/optimal_trajectory_generator.git
cd optimal_trajectory_generator
catkin build
```

Remember to source the `setup.bash`.

### 2. Play with this

In one terminal, run

```
roslaunch traj_generator demo.launch 
```

This will launch Rviz and config it properly.

Then, in another terminal, run

```
rosrun traj_generator main_node.py 
```

Now use the `2D Nav Goal` to set a few waypoints in RViz. When the number of waypoints reaches the predefined number (default is 8), trajectory generation will be triggered automatically and you can see the trajectory with its dynamic.

## Reference

[1] Z. Wang, X. Zhou, C. Xu and F. Gao, "Geometrically Constrained Trajectory Optimization for Multicopters," in IEEE Transactions on Robotics, vol. 38, no. 5, pp. 3259-3278, Oct. 2022, doi: 10.1109/TRO.2022.3160022.

