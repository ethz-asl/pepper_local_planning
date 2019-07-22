from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
import clustering

class Responsive(object):
    def __init__(self):
        # consts
        self.kLidarTopic = "/combined_scan"
        self.kMaxObstacleVel_ms = 10. # [m/s]
        # vars
        self.msg_prev = None
        # ROS
        rospy.init_node('responsive', anonymous=True)
        rospy.Subscriber(self.kLidarTopic, LaserScan, self.scan_callback)
        self.pubs = [rospy.Publisher("debug{}".format(i), LaserScan, queue_size=1) for i in range(3)]
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.signal_shutdown('KeyboardInterrupt')


    def scan_callback(self, msg):
        # edge case: first callback
        if self.msg_prev is None:
            self.msg_prev = msg
            return

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
        clusters, x, y = clustering.euclidean_clustering(s_next, 0.05)
        cluster_sizes = clustering.cluster_sizes(len(s_next), clusters)
        s_next[cluster_sizes <= 3] = 25


        if False:
            plt.ion()
            plt.figure(1, figsize=(20,5))
            plt.cla()
            plt.plot(s_next)
            plt.plot(np.diff(s_next))
            plt.plot(peaks)
            plt.show()
            plt.pause(0.001)



        msg_next = deepcopy(msg)
        msg_next.ranges = s_next
        # for pretty colors
        cluster_ids = clustering.cluster_ids(len(x), clusters)
        random_map = np.arange(len(cluster_ids))
        np.random.shuffle(random_map)
        cluster_ids = random_map[cluster_ids]
        msg_next.intensities = cluster_ids
        self.pubs[0].publish(msg_next)

        msg_prev = deepcopy(msg)
        msg_prev.ranges = self.msg_prev.ranges
        self.pubs[1].publish(msg_prev)

        # ...

        # finally, set up for next callback
        self.msg_prev = msg

if __name__=="__main__":
    responsive = Responsive()
