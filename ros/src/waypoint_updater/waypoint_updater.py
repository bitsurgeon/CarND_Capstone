#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

from scipy.spatial import KDTree
from std_msgs.msg import Int32
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

# * Number of waypoints we will publish, change to smaller for smoother simulator experience
LOOKAHEAD_WPS = 80
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.stopline_wp_idx = None
        
        # rospy.spin()
        self.loop() # replace the above rospy.spin() to get flexibility for rate control

    def loop(self):
        # * waypoints publishing rate, change to smaller for smoother simulator experience
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints_tree:
                # get the closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        # Positive value of the dot means the closest waypoint is behind the car
        # then we take the next waypoint
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx

    def publish_waypoints(self, closest_idx):
        # # ignore the traffic light
        # lane = Lane()
        # lane.header = self.base_waypoints.header
        # # take the required waypoints ahead of the car
        # lane.waypoints = self.base_waypoints.waypoints[closest_idx : closest_idx+LOOKAHEAD_WPS]
        # self.final_waypoints_pub.publish(lane)

        # obey the traffic light
        if self.stopline_wp_idx:
            final_lane = self.generate_lane(closest_idx)
            self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            # if no red light or the red light is beyond the end of the planned trajectory, then follow the planned trajectory
            lane.waypoints = base_waypoints
        else:
            # if red light, then brake to the stop line
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        # calculate the waypoints for deceleration
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            # calculate in how many waypoints, the car need to stop
            stop_idx = max(self.stopline_wp_idx - closest_idx - 3, 0) # 3 waypoints back for the car nose stop before the stop line.

            # the distance between the waypoint to the stop waypoint
            dist = self.distance(waypoints, i, stop_idx)

            # calculate the velocity to achieve 0 velocity, v**2 = 2*a*s
            vel = math.sqrt(2 * MAX_DECEL * dist)

            if vel < 1.0:
                vel = 0.

            # limit possible very high velocity to speed limit
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        # store the PoseStamped received
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # store the base wayppoints received, done only once
        self.base_waypoints = waypoints
        
        # store these waypoints in KDTree for efficient search of closest waypoint later
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message.
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
