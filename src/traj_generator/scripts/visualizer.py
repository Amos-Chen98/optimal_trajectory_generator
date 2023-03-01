'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-01 20:47:55
'''

import rospy
import numpy as np
from matplotlib import cm
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class Visualizer():

    def get_path(self, pos_array, vel_array):
        path = MarkerArray()
        vel_max = np.max(vel_array)
        color_codes = (vel_array/vel_max*256).astype(int)

        for i in range(len(pos_array)-1):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.seq = i
            marker.header.stamp = rospy.get_rostime()
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            line_head = [pos_array[i][0], pos_array[i][1], pos_array[i][2]]
            line_tail = [pos_array[i+1][0], pos_array[i+1][1], pos_array[i+1][2]]
            marker.points = [Point(line_head[0], line_head[1], line_head[2]), Point(line_tail[0], line_tail[1], line_tail[2])]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            color = cm.jet(color_codes[i])
            marker.color = ColorRGBA(color[0], color[1], color[2], color[3])

            path.markers.append(marker)

        return path

    def get_marker_array(self, pos_array):
        '''
        Get marker array from pos array
        input: pos_array - np.ndarray of (n,3)
        output: markerArray - visualization_msgs.msg.MarkerArray
        '''
        markerArray = MarkerArray()

        for i in range(len(pos_array)):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.seq = i
            marker.header.stamp = rospy.get_rostime()
            marker.id = i

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = pos_array[i][0]
            marker.pose.position.y = pos_array[i][1]
            marker.pose.position.z = pos_array[i][2]

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.4
            marker.scale.y = 0.4
            marker.scale.z = 0.4

            color = cm.jet(int(i*1.0/len(pos_array)*256))
            marker.color = ColorRGBA(color[0], color[1], color[2], color[3])

            markerArray.markers.append(marker)

        return markerArray
