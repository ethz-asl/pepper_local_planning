from copy import deepcopy
import numpy as np
import rospy
import tf
import threading
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from timeit import default_timer as timer
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

# local packages
from pose2d import Pose2D, apply_tf, apply_tf_to_vel, apply_tf_to_pose, inverse_pose2d
import clustering
import dynamic_window

PLAN_EVEN_IF_STOPPED = False

class Responsive(object):
    def __init__(self, args):
        self.args = args
        # consts
        self.kLidarTopic = "/combined_scan"
        self.kFixedFrameTopic = "/pepper_robot/odom"
        self.kCmdVelTopic = "/cmd_vel"
        self.kGesturesTopic = "/gestures"
        self.kGlobalWaypointTopic = "/global_planner/current_waypoint"
        self.kFixedFrame = "odom"  # ideally something truly fixed, like /map
        self.kRobotFrame = "base_footprint"
        self.kMaxObstacleVel_ms = 10.  # [m/s]
        self.kRobotComfortRadius_m = rospy.get_param("/robot_comfort_radius", 0.7)
        self.kRobotRadius_m = rospy.get_param("/robot_radius", 0.3)
        self.kGesturesCooldownTime = 3.  # seconds
        # vars
        self.msg_prev = None
        self.odom = None
        self.tf_rob_in_fix = None
        self.tf_goal_in_fix = None
        self.lock = threading.Lock()  # for avoiding race conditions
        self.STOP = True  # disables autonomous control
        if args.no_stop:
            self.STOP = False
        self.GESTURES = False
        if args.gestures:
            self.GESTURES = True
        self.is_tracking_global_path = False
        # ROS
        rospy.init_node('responsive', anonymous=True)
        rospy.Subscriber(self.kGlobalWaypointTopic, Marker, self.global_waypoint_callback, queue_size=1)
        rospy.Subscriber(self.kLidarTopic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.kFixedFrameTopic, Odometry, self.odom_callback, queue_size=1)
        try:
            from frame_msgs.msg import TrackedPersons
            rospy.Subscriber("/rwth_tracker/tracked_persons", TrackedPersons,
                             self.trackedpersons_callback, queue_size=1)
        except ImportError:
            pass
        self.pubs = [rospy.Publisher("debug{}".format(i), LaserScan, queue_size=1) for i in range(3)]
        self.cmd_vel_pub = rospy.Publisher(self.kCmdVelTopic, Twist, queue_size=1)
        self.gestures_pub = rospy.Publisher(self.kGesturesTopic, String, queue_size=1)
        # tf
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        # Timers
        rospy.Timer(rospy.Duration(0.001), self.tf_callback)
        # Services
        rospy.Service('stop_autonomous_motion', Trigger,
                      self.stop_autonomous_motion_service_call)
        rospy.Service('resume_autonomous_motion', Trigger,
                      self.resume_autonomous_motion_service_call)
        rospy.Service('enable_gestures', Trigger,
                      self.enable_gestures_service_call)
        rospy.Service('disable_gestures', Trigger,
                      self.disable_gestures_service_call)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            # publish cmd_vel
            cmd_vel_msg = Twist()
            self.cmd_vel_pub.publish(cmd_vel_msg)
            rospy.signal_shutdown('KeyboardInterrupt')

    def odom_callback(self, msg):
        self.odom = msg

    def tf_callback(self, event=None):
        try:
            self.tf_rob_in_fix = self.tf_listener.lookupTransform(
                self.kFixedFrame, self.kRobotFrame, rospy.Time(0)
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
        # periodically publish goal tf
        if self.tf_goal_in_fix is not None:
            self.tf_br.sendTransform(
                self.tf_goal_in_fix[0],
                self.tf_goal_in_fix[1],
                rospy.Time.now(),
                "goal",
                self.kFixedFrame,
            )

    def global_waypoint_callback(self, msg):
        """ If a global path is received (in map frame), try to track it """
        with self.lock:
            waypoint_in_msg = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
            ])
            # msg frame to fixed frame
            if self.kFixedFrame != msg.header.frame_id:
                try:
                    tf_msg_in_fix = self.tf_listener.lookupTransform(
                        self.kFixedFrame,
                        msg.header.frame_id,
                        msg.header.stamp - rospy.Duration(0.1),
                    )
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    print("Could not find transform from waypoint frame to fixed.")
                    print(e)
                    return
                waypoint_in_fix = apply_tf(waypoint_in_msg, Pose2D(tf_msg_in_fix))
            else:
                waypoint_in_fix = waypoint_in_msg
            self.tf_goal_in_fix = (np.array([waypoint_in_fix[0], waypoint_in_fix[1], 0.]),  # trans
                                   tf.transformations.quaternion_from_euler(0, 0, 0))  # quat
            rospy.loginfo("Responsive: waypoint received and set.")

    def scan_callback(self, msg):
        atic = timer()
        # edge case: first callback
        if self.msg_prev is None:
            self.msg_prev = msg
            return
        if self.odom is None:
            print("odom not received yet")
            return
        if self.tf_rob_in_fix is None:
            print("tf_rob_in_fix not found yet")
            return
        if self.tf_goal_in_fix is None:
            self.tf_goal_in_fix = self.tf_rob_in_fix
            print("responsive: waypoint set to robot position (scan callback)")
        if self.STOP and not PLAN_EVEN_IF_STOPPED:
            return
        # TODO check that odom and tf are not old

        # measure rotation TODO
        s = np.array(msg.ranges)

        # prediction
        dt = (msg.header.stamp - self.msg_prev.header.stamp).to_sec()
        s_prev = np.array(self.msg_prev.ranges)
        ds = (s - s_prev)
        max_ds = self.kMaxObstacleVel_ms * dt
        ds_capped = ds
        ds_capped[np.abs(ds) > max_ds] = 0
        s_next = np.maximum(0, s + ds_capped).astype(np.float32)

        # cluster
        EUCLIDEAN_CLUSTERING_THRESH_M = 0.05
        angles = np.linspace(0, 2 * np.pi, s_next.shape[0] + 1, dtype=np.float32)[:-1]
        clusters, x, y = clustering.euclidean_clustering(s_next, angles,
                                                         EUCLIDEAN_CLUSTERING_THRESH_M)
        cluster_sizes = clustering.cluster_sizes(len(s_next), clusters)
        s_next[cluster_sizes <= 3] = 25

        # dwa
        # Get state
        # goal in robot frame
        goal_in_robot_frame = apply_tf_to_pose(
            Pose2D(self.tf_goal_in_fix), inverse_pose2d(Pose2D(self.tf_rob_in_fix)))
        gx = goal_in_robot_frame[0]
        gy = goal_in_robot_frame[1]

        # robot speed in robot frame
        u = self.odom.twist.twist.linear.x
        v = self.odom.twist.twist.linear.y
#         w = self.odom.twist.twist.angular.z

        DWA_DT = 0.5
        COMFORT_RADIUS_M = self.kRobotComfortRadius_m
        MAX_XY_VEL = 0.5
#         tic = timer()
        best_u, best_v, best_score = dynamic_window.linear_dwa(
            s_next,
            angles,
            u, v, gx, gy, DWA_DT,
            DV=0.05,
            UMIN=0 if self.args.forward_only else -MAX_XY_VEL,
            UMAX=MAX_XY_VEL,
            VMIN=-MAX_XY_VEL,
            VMAX=MAX_XY_VEL,
            AMAX=10.,
            COMFORT_RADIUS_M=COMFORT_RADIUS_M,
        )
#         toc = timer()
#         print(best_u * DWA_DT, best_v * DWA_DT, best_score)
#         print("DWA: {:.2f} Hz".format(1/(toc-tic)))

        # Slow turn towards goal
        # TODO
        best_w = 0
        WMAX = 0.5
        angle_to_goal = np.arctan2(gy, gx)  # [-pi, pi]
        if np.sqrt(gx * gx + gy * gy) > 0.5:  # turn only if goal is far away
            if np.abs(angle_to_goal) > (np.pi / 4 / 10):  # deadzone
                best_w = np.clip(angle_to_goal, -WMAX, WMAX)  # linear ramp

        if not self.STOP:
            # publish cmd_vel
            cmd_vel_msg = Twist()
            SIDEWAYS_FACTOR = 0.3 if self.args.forward_only else 1.
            cmd_vel_msg.linear.x = np.clip(best_u * 0.5, -0.3, 0.3)
            cmd_vel_msg.linear.y = np.clip(best_v * 0.5 * SIDEWAYS_FACTOR, -0.3, 0.3)
            cmd_vel_msg.angular.z = best_w
            self.cmd_vel_pub.publish(cmd_vel_msg)

        # publish cmd_vel vis
        pub = rospy.Publisher("/dwa_cmd_vel", Marker, queue_size=1)
        mk = Marker()
        mk.header.frame_id = self.kRobotFrame
        mk.ns = "arrows"
        mk.id = 0
        mk.type = 0
        mk.action = 0
        mk.scale.x = 0.02
        mk.scale.y = 0.02
        mk.color.b = 1
        mk.color.a = 1
        mk.frame_locked = True
        pt = Point()
        pt.x = 0
        pt.y = 0
        pt.z = 0.03
        mk.points.append(pt)
        pt = Point()
        pt.x = 0 + best_u * DWA_DT
        pt.y = 0 + best_v * DWA_DT
        pt.z = 0.03
        mk.points.append(pt)
        pub.publish(mk)
        pub = rospy.Publisher("/dwa_goal", Marker, queue_size=1)
        mk = Marker()
        mk.header.frame_id = self.kRobotFrame
        mk.ns = "arrows"
        mk.id = 0
        mk.type = 0
        mk.action = 0
        mk.scale.x = 0.02
        mk.scale.y = 0.02
        mk.color.g = 1
        mk.color.a = 1
        mk.frame_locked = True
        pt = Point()
        pt.x = 0
        pt.y = 0
        pt.z = 0.03
        mk.points.append(pt)
        pt = Point()
        pt.x = 0 + gx
        pt.y = 0 + gy
        pt.z = 0.03
        mk.points.append(pt)
        pub.publish(mk)
        pub = rospy.Publisher("/dwa_radius", Marker, queue_size=1)
        mk = Marker()
        mk.header.frame_id = self.kRobotFrame
        mk.ns = "radius"
        mk.id = 0
        mk.type = 3
        mk.action = 0
        mk.pose.position.z = -0.1
        mk.scale.x = COMFORT_RADIUS_M * 2
        mk.scale.y = COMFORT_RADIUS_M * 2
        mk.scale.z = 0.01
        mk.color.b = 1
        mk.color.g = 1
        mk.color.r = 1
        mk.color.a = 1
        mk.frame_locked = True
        pub.publish(mk)

        # publish scan prediction
        msg_next = deepcopy(msg)
        msg_next.ranges = s_next
        # for pretty colors
        cluster_ids = clustering.cluster_ids(len(x), clusters)
        random_map = np.arange(len(cluster_ids))
        np.random.shuffle(random_map)
        cluster_ids = random_map[cluster_ids]
        msg_next.intensities = cluster_ids
        self.pubs[0].publish(msg_next)

        # publish past
        msg_prev = deepcopy(msg)
        msg_prev.ranges = self.msg_prev.ranges
        self.pubs[1].publish(msg_prev)

        # ...

        # finally, set up for next callback
        self.msg_prev = msg

        atoc = timer()
        if self.args.hz:
            print("DWA callback: {:.2f} Hz".format(1 / (atoc - atic)))

    def trackedpersons_callback(self, msg):
        try:
            tf_msg_in_rob = self.tf_listener.lookupTransform(
                self.kRobotFrame, msg.header.frame_id, msg.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return
        p2_msg_in_rob = Pose2D(tf_msg_in_rob)
        for track in msg.tracks:
            pose = [track.pose.pose.position.x, track.pose.pose.position.y]
            pose_in_rob = apply_tf(np.array([pose]), p2_msg_in_rob)[0]
            vel = [track.twist.twist.linear.x, track.twist.twist.linear.y, 0]
            vel_in_rob = apply_tf_to_vel(np.array(vel), p2_msg_in_rob)
            is_in_front = pose_in_rob[0] > 0
            is_close = np.linalg.norm(pose_in_rob[:2]) <= 2.
            is_static = np.linalg.norm(vel_in_rob[:2]) <= 0.5
            if is_in_front and is_close and is_static:
                if not self.STOP and self.GESTURES:
                    self.gestures_pub.publish(String("animations/Stand/Gestures/You_2"))
                    rospy.sleep(self.kGesturesCooldownTime)
                    self.gestures_pub.publish(String("animations/Stand/Gestures/Desperate_4"))
                    return

    def enable_gestures_service_call(self, req):
        with self.lock:
            if not self.GESTURES:
                rospy.loginfo("Enabling gestures.")
            self.GESTURES = True
        return TriggerResponse(True, "")

    def disable_gestures_service_call(self, req):
        with self.lock:
            if self.GESTURES:
                rospy.loginfo("Disabling gestures.")
            self.GESTURES = False
        return TriggerResponse(True, "")

    def stop_autonomous_motion_service_call(self, req):
        with self.lock:
            if not self.STOP:
                print("Surrendering robot control")
                cmd_vel_msg = Twist()
                self.cmd_vel_pub.publish(cmd_vel_msg)
            self.STOP = True
        return TriggerResponse(True, "")

    def resume_autonomous_motion_service_call(self, req):
        with self.lock:
            if self.STOP:
                print("Assuming robot control")
                # re set goal
                if self.tf_rob_in_fix is None:
                    print("couldn't reset goal: tf_rob_in_fix not found yet")
                else:
                    self.tf_goal_in_fix = self.tf_rob_in_fix
                    print("responsive: waypoint set to current position (assumed control)")
            self.STOP = False
        return TriggerResponse(True, "")


def parse_args():
    import argparse
    # Arguments
    parser = argparse.ArgumentParser(description='Responsive motion planner for pepper')
    parser.add_argument(
        '--no-stop',
        action='store_true',
        help='if set, the planner will immediately send cmd_vel instead of waiting for hand-over',
    )
    parser.add_argument(
        '--hz',
        action='store_true',
        help='if set, prints planner frequency to script output',
    )
    parser.add_argument(
        '--forward-only',
        action='store_true',
        help='if set, the DWA planner is only allowed to move forwards',
    )
    parser.add_argument(
        '--gestures',
        action='store_true',
        help='if set, gestures are enabled from the start',
    )

    ARGS, unknown_args = parser.parse_known_args()

    # deal with unknown arguments
    # ROS appends some weird args, ignore those, but not the rest
    if unknown_args:
        non_ros_unknown_args = rospy.myargv(unknown_args)
        if non_ros_unknown_args:
            print("unknown arguments:")
            print(non_ros_unknown_args)
            parser.parse_args(args=["--help"])
            raise ValueError
    return ARGS


if __name__ == "__main__":
    args = parse_args()
    responsive = Responsive(args)
