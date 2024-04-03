'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2024-04-03 15:58:19
'''
import os
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)
from matplotlib import pyplot as plt
from visualizer import Visualizer
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from traj_planner import MinJerkPlanner
import time
from visualization_msgs.msg import MarkerArray


class TrajGenerator():
    def __init__(self, node_name="traj_generator"):
        # Node
        rospy.init_node(node_name)

        self.planner = MinJerkPlanner()
        self.visualizer = Visualizer()

        # Customized parameters
        self.total_wpts_num = 8  # include the init and final wpts
        self.target_wpts = np.zeros([self.total_wpts_num, 3])
        self.get_wpts_num = 0  # current number of wpts received
        self.T = 5  # time for each segment

        # Subscribers
        self.target_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.record_wpt)

        # Publishers
        self.des_wpts_pub = rospy.Publisher('/des_wpts', MarkerArray, queue_size=10)
        self.des_path_pub = rospy.Publisher('/des_path', MarkerArray, queue_size=10)

        rospy.loginfo("Traj_generator initialized!")

    def record_wpt(self, point):
        '''
        Record the desired waypoints
        '''
        self.get_wpts_num += 1
        rospy.loginfo("Waypoint added {}/{}: [{}, {}, {}]".format(self.get_wpts_num,
                                                                  self.total_wpts_num,
                                                                  point.pose.position.x,
                                                                  point.pose.position.y,
                                                                  point.pose.position.z))
        wpt_pos = np.array([point.pose.position.x, point.pose.position.y, point.pose.position.z])
        self.target_wpts[self.get_wpts_num-1, :] = wpt_pos
        self.visualize_des_wpts(self.target_wpts[:self.get_wpts_num])
        if self.get_wpts_num == self.total_wpts_num:
            self.get_wpts_num = 0
            self.gen_traj()

    def gen_traj(self):
        head_state = np.zeros([3, 3])  # desired [pos, vel, acc]
        tail_state = np.zeros([3, 3])
        head_state[0] = self.target_wpts[0]  # specify the initial position
        tail_state[0] = self.target_wpts[-1]
        int_wpts = self.target_wpts[1:-1]
        ts = self.T * np.ones(len(int_wpts)+1)  # you can change this to other time allocation method

        start_time = time.time()
        self.planner.plan(3, head_state, tail_state, int_wpts, ts)
        end_time = time.time()

        rospy.loginfo(f"Trajectory generated. Time cost: {end_time - start_time}s")

        self.visualize_des_path()
        self.plot_state_curve()

    def visualize_des_wpts(self, wpts):
        '''
        Visualize the desired waypoints as markers
        '''
        des_wpts = self.visualizer.get_marker_array(wpts)
        self.des_wpts_pub.publish(des_wpts)

    def visualize_des_path(self):
        '''
        Visualize the desired path, where high-speed pieces and low-speed pieces are colored differently
        '''
        pos_array = self.planner.get_pos_array()
        if pos_array.shape[1] == 2:
            pos_array = np.hstack((pos_array, np.zeros([len(pos_array), 1])))
        vel_array = np.linalg.norm(self.planner.get_vel_array(), axis=1)  # shape: (n,)
        des_path = self.visualizer.get_path(pos_array, vel_array)
        self.des_path_pub.publish(des_path)

    def plot_state_curve(self):
        # delete all existing plots
        plt.close('all')
        final_ts = self.planner.ts
        t_samples = np.arange(0, sum(final_ts), 0.1)
        t_cum_array = np.cumsum(final_ts)
        vel = self.planner.get_vel_array()
        acc = self.planner.get_acc_array()
        jer = self.planner.get_jer_array()
        snap = self.planner.get_snap_array()

        # get the norm of vel, acc and jer
        vel_norm = np.linalg.norm(vel, axis=1)
        acc_norm = np.linalg.norm(acc, axis=1)
        jer_norm = np.linalg.norm(jer, axis=1)
        snap_norm = np.linalg.norm(snap, axis=1)

        plt.figure("X axis states")
        plt.plot(t_samples, vel[:, 0], label='Vel_x ($m\cdot s^{-1}$)')
        plt.plot(t_samples, acc[:, 0], label='Acc_x ($m\cdot s^{-2}$)')
        plt.plot(t_samples, jer[:, 0], label='Jerk_x ($m\cdot s^{-3}$)')
        plt.plot(t_samples, snap[:, 0], label='Snap_x ($m\cdot s^{-4}$)')
        plt.xlabel('t/s')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

        plt.figure("Y axis states")
        plt.plot(t_samples, vel[:, 1], label='Vel_y ($m\cdot s^{-1}$)')
        plt.plot(t_samples, acc[:, 1], label='Acc_y ($m\cdot s^{-2}$)')
        plt.plot(t_samples, jer[:, 1], label='Jerk_y ($m\cdot s^{-3}$)')
        plt.plot(t_samples, snap[:, 1], label='Snap_y ($m\cdot s^{-4}$)')
        plt.xlabel('t/s')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

        plt.figure("States norm")
        plt.plot(t_samples, vel_norm, label='Vel ($m\cdot s^{-1}$)')
        plt.plot(t_samples, acc_norm, label='Acc ($m\cdot s^{-2}$)')
        plt.plot(t_samples, jer_norm, label='Jerk ($m\cdot s^{-3}$)')
        plt.plot(t_samples, snap_norm, label='Snap ($m\cdot s^{-4}$)')
        # plt.vlines(t_cum_array, 0, np.max(vel_norm))  # mark the int_wpts
        plt.xlabel('t/s')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()

        plt.show()


if __name__ == '__main__':
    traj_generator = TrajGenerator()
    rospy.spin()
