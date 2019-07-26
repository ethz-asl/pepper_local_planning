from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import rospy
import tf
import threading
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from timeit import default_timer as timer
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist
from std_srvs.srv import Trigger, TriggerResponse

# local packages
from pose2d import Pose2D, apply_tf_to_pose, inverse_pose2d
import clustering
import dynamic_window

class Responsive(object):
    def __init__(self):
        # consts
        self.kLidarTopic = "/combined_scan"
        self.kFixedFrameTopic = "/pepper_robot/odom"
        self.kCmdVelTopic = "/cmd_vel"
        self.kFixedFrame = "odom" # ideally something truly fixed, like /map
        self.kRobotFrame = "base_footprint"
        self.kMaxObstacleVel_ms = 10. # [m/s]
        # vars
        self.msg_prev = None
        self.odom = None
        self.tf_rob_in_fix = None
        self.tf_goal_in_fix = None
        self.lock = threading.Lock() # for avoiding race conditions
        self.STOP = True # disables autonomous control
        # ROS
        rospy.init_node('responsive', anonymous=True)
        rospy.Subscriber(self.kLidarTopic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.kFixedFrameTopic, Odometry, self.odom_callback, queue_size=1)
        self.pubs = [rospy.Publisher("debug{}".format(i), LaserScan, queue_size=1) for i in range(3)]
        self.cmd_vel_pub = rospy.Publisher(self.kCmdVelTopic, Twist, queue_size=1)
        # tf
        self.tf_listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        # Timers
        rospy.Timer(rospy.Duration(0.001), self.tf_callback)
        # Services
        rospy.Service('stop_autonomous_motion', Trigger, 
                self.stop_autonomous_motion_service_call)
        rospy.Service('resume_autonomous_motion', Trigger, 
                self.resume_autonomous_motion_service_call)
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
             self.tf_rob_in_fix = self.tf_listener.lookupTransform(self.kFixedFrame, self.kRobotFrame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            return
        # periodically publish goal tf
        if self.tf_goal_in_fix is not None:
            self.br.sendTransform(self.tf_goal_in_fix[0],
                             self.tf_goal_in_fix[1],
                             rospy.Time.now(),
                             "goal",
                             self.kFixedFrame)

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
            print("goal set")
        # TODO check that odom and tf are not old

        # measure rotation TODO
        s = np.array(msg.ranges)

        # prediction
        dt = (msg.header.stamp - self.msg_prev.header.stamp).to_sec()
        s_prev =  np.array(self.msg_prev.ranges)
        ds = (s - s_prev)
        max_ds = self.kMaxObstacleVel_ms * dt
        ds_capped = ds
        ds_capped[np.abs(ds) > max_ds] = 0
        s_next = np.maximum(0, s + ds_capped).astype(np.float32)

        # cluster
        EUCLIDEAN_CLUSTERING_THRESH_M = 0.05
        angles = np.linspace(0, 2*np.pi, s_next.shape[0]+1, dtype=np.float32)[:-1]
        clusters, x, y = clustering.euclidean_clustering(s_next, angles,
                                                         EUCLIDEAN_CLUSTERING_THRESH_M)
        cluster_sizes = clustering.cluster_sizes(len(s_next), clusters)
        s_next[cluster_sizes <= 3] = 25


        # dwa
        # Get state
        # goal in robot frame
        goal_in_robot_frame = apply_tf_to_pose(Pose2D(self.tf_goal_in_fix), inverse_pose2d(Pose2D(self.tf_rob_in_fix)))
        gx = goal_in_robot_frame[0]
        gy = goal_in_robot_frame[1]
        
        # robot speed in robot frame
        u = self.odom.twist.twist.linear.x
        v = self.odom.twist.twist.linear.y
        w = self.odom.twist.twist.angular.z


        DWA_DT = 0.5
        tic = timer()
        best_u, best_v, best_score = dynamic_window.linear_dwa(s_next,
            angles,
            u, v, gx, gy, DWA_DT,
            DV=0.05,
            UMIN=-0.5,
            UMAX=0.5,
            VMIN=-0.5,
            VMAX=0.5,
            AMAX=10.,
            COMFORT_RADIUS_M=0.5,
            )
        toc = timer()
        print(best_u * DWA_DT, best_v * DWA_DT, best_score)
        print("DWA: {:.2f} Hz".format(1/(toc-tic)))

        if not self.STOP:
            # publish cmd_vel
            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = best_u * 0.8
            cmd_vel_msg.linear.y = best_v * 0.8
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
        print("DWA callback: {:.2f} Hz".format(1/(atoc-atic)))

    def stop_autonomous_motion_service_call(self, req):
        with self.lock:
            if not self.STOP:
                print("Surrendering robot control")
            self.STOP = True
        return TriggerResponse(True, "")

    def resume_autonomous_motion_service_call(self, req):
        with self.lock:
            if self.STOP:
                print("Assuming robot control")
            self.STOP = False
        return TriggerResponse(True, "")

if __name__=="__main__":
    responsive = Responsive()
