import roslib; roslib.load_manifest('project4')
import rospy
import rospkg
import tf
import transform2d
import numpy
import sys
import maze
import going_further

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from kobuki_msgs.msg import SensorState

# we will run our controller at 100Hz
CONTROL_PERIOD = rospy.Duration(0.01)

# one degree is pi/180 radians
DEG = numpy.pi / 180.0

# one foot is 0.3048 meters
FT = 0.3048

ANGULAR_TOL = 1*DEG # rad 
ANGULAR_RAMP_TIME = 2.0 # s
ANGULAR_RAMP_SPEED = 1.5 # rad/s
ANGULAR_MIN_SPEED = 0.1 # rad/s
ANGULAR_GAIN = 3.0 # converts rad to rad/s

LINEAR_TOL = 0.005 # m 
LINEAR_MIN_SPEED = 0.02 # m/s
LINEAR_RAMP_TIME = 4.0 # s
LINEAR_RAMP_SPEED = 1.0 # m/s
LINEAR_GAIN = 3.0 # converts m to m/s

# given a Maze() class instance m, as well as a starting point (x0,
# y0), an initial direction dir, and a goal point, return a list of
# maze commands (e.g. 'forward', 'turnleft', 'turnright', ...)
# to solve the maze

#My work
def solve_maze(m, x0, y0, dir, x1, y1):
    print(x0, y0, x1, y1)
    path = maze.find_path(m, x0, y0, x1, y1)
    print(path)
    cmd = follow_path(path, dir)
    return cmd 

def follow_path(path,init_dir):
    cmd = []
    if init_dir == 1:
        #cmd.append('turnleft')
        current_dir = 'right'
    elif init_dir == 0:
        current_dir = 'left'
        #cmd.append('turnright')
    elif init_dir == 3:
        #cmd.append('turnright')
        #cmd.append('turnright')
        current_dir = 'backward'
    else:
        current_dir = "forward"
   
    if path == None: 
        print("No viable path towards life's success")
        return []
    print(path)  
    for i in range(len(path)-1):
        xcur, ycur = path[i][0], path[i][1]
        xnext, ynext = path[i+1][0], path[i+1][1]
        if current_dir == 'forward':

                if xnext - xcur == 1: 
                    cmd.append("turnright")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "right"
                elif xnext - xcur == -1:
                    cmd.append("turnleft")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "left"
                else:
                    cmd.append("forward")
                    cmd.append("nudge")

        elif current_dir == 'left':
                if ynext - ycur == 1:
                    cmd.append("turnright")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "forward"

                elif ynext - ycur == -1:
                    cmd.append("turnleft")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "backward"
                else:
                    cmd.append("forward")
                    cmd.append("nudge")
                
        elif current_dir == 'right':
                if ynext - ycur == 1:
                    cmd.append("turnleft")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "forward"
                elif ynext - ycur == -1:
                    cmd.append("turnright")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "backward"
                else:
                    cmd.append("forward")
                    cmd.append("nudge")
         
        else: 
                if xnext - xcur == 1:
                    cmd.append("turnleft")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "right"
                elif xnext - xcur == -1:
                    cmd.append("turnright")
                    cmd.append("forward")
                    cmd.append("nudge")
                    current_dir = "left"
                else:
                    cmd.append("forward")
                    cmd.append("nudge")
    return cmd

# Code from secondary source
class Controller:

    # Initializer
    def __init__(self):

        # Setup this node
        rospy.init_node('Controller')

        # Create a TransformListener - we will use it both to fetch
        # odometry readings and to get the transform between the
        # Kinect's depth image and the base frame.
        self.tf_listener = tf.TransformListener()

        # Stash the 2D transformation from depth image to base frame.
        self.base_from_depth = None

        # Let's start out in the initializing state
        self.reset_state('initializing')

        # These member variables will be set by the laser scan
        # callback.
        self.angles = None
        self.ranges = None
        self.points = None

        # For safety
        self.should_stop = False
        self.stop_time = rospy.get_rostime() - rospy.Duration(100.0)

        # Set up maze
        self.maze_commands = []
        self.command_index = -1
        self.maze = None

        args = rospy.myargv(argv=sys.argv)[1:]

        if args[0] == 'solve':
            
            rospack = rospkg.RosPack()
            project4_path = rospack.get_path('project4')
            maze_file = project4_path + '/data/' + args[1]
            x0, y0, dir0, x1, y1 = maze.split_command(args[2:])
            m = maze.Maze()
            m.load(maze_file)

            self.maze_commands = solve_maze(m, x0, y0, dir0, x1, y1)

        else:
            
            self.maze_commands = args
            
        rospy.loginfo('maze_commands: ' + str(self.maze_commands))

        # Create a publisher for commanded velocity
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/commands/velocity',
                                           Twist, queue_size=10)

        # Subscribe to laser scan data
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # Subscribe to bumpers/cliff sensors
        rospy.Subscriber('/mobile_base/sensors/core',
                         SensorState, self.sensor_callback)

        # Call the controll callback at 100 HZ
        rospy.Timer(CONTROL_PERIOD, self.control_callback)


    # Set the current state to the state given.
    #
    # You can pass reset_time=False to disable resetting the
    # self.start_time variable. This is useful if you want to switch
    # from one behavior to another (i.e. "turnright" to "straighten")
    # without restricting speed.
    def reset_state(self, state, reset_time=True):

        self.state = state
        self.start_pose = None

        if reset_time:
            self.start_time = rospy.get_rostime()
            self.labeled_state = state

        if state == 'initializing':
            rospy.loginfo('waiting for TF and laser scan...')
            
    # Called whenever sensor messages are received.
    def sensor_callback(self, msg):

        if (msg.cliff or msg.bumper):
            self.should_stop = True
            self.stop_time = rospy.get_rostime()
        else:
            self.should_stop = False


    # Set up the transformation from depth camera to base frame.
    # We will need this later to deal with laser scanner.
    def setup_tf(self):

        # We might need to do this more than once because the
        # TransformListener might not be ready yet.
        try:

            ros_xform = self.tf_listener.lookupTransform(
                '/base_footprint', '/camera_depth_frame',
                rospy.Time(0))

        except tf.LookupException:

            return False

        self.base_from_depth = \
            transform2d.transform2d_from_ros_transform(ros_xform)

        return True

    # Called whenever we hear from laser scanner. This just sets up
    # the self.angles, self.ranges, and self.points member variables.
    def scan_callback(self, msg):

        # Don't do anything til we have the transform from depth
        # camera to base frame.
        if self.base_from_depth is None:
            if not self.setup_tf():
                return

        # Get # of range returns
        count = len(msg.ranges)

        # Create angle array of size N
        self.angles = (msg.angle_min +
                       numpy.arange(count, dtype=float) * msg.angle_increment)

        # Create numpy array from range returns (note many could be
        # NaN indicating no return for a given angle).
        self.ranges = numpy.array(msg.ranges)

        # Points is a 2xN array of cartesian points in depth camera frame
        pts = self.ranges * numpy.vstack( ( numpy.cos(self.angles),
                                            numpy.sin(self.angles) ) )

        # This converts points from depth camera frame to base frame
        # and reshapes into an Nx2 array so that self.points[i] is the
        # point corresponding to self.angles[i].
        self.points = self.base_from_depth.transform_fwd(pts).T

    # Given a list of desired angles (e.g. [-5*DEG, 5*DEG], look up
    # the indices of the closest valid ranges to those angles. If
    # there is no valid range within the cutoff angular distance,
    # returns None. 
    def lookup_angles(self, desired_angles, cutoff=3*DEG):

        # Don't return anything if no data.
        if self.angles is None:
            return None

        # Get indices of all non-NaN ranges
        ok_idx = numpy.nonzero(~numpy.isnan(self.ranges))[0]

        # Check all NaN
        if not len(ok_idx):
            return None

        # Build up array of indices to return
        indices = []

        # For each angle passed in
        for angle in desired_angles:

            # Find the closest index
            angle_err = numpy.abs(angle - self.angles[ok_idx])
            i = angle_err.argmin()

            # If too far away from desired, fail :(
            if angle_err[i] > cutoff:
                return None

            # Append index of closest
            indices.append(ok_idx[i])

        # Return the array we built up
        return indices

    # Look up points at the angles given (see lookup_angles for
    # interpretation of desired_angles, cutoff). 
    def points_at_angles(self, desired_angles, cutoff=3*DEG):

        indices = self.lookup_angles(desired_angles, cutoff)
        if indices is None:
            return None

        return self.points[indices]

    # Gets the current pose of the robot w.r.t. odometry frame.
    def get_current_pose(self):

        try:
            ros_xform = self.tf_listener.lookupTransform(
                '/odom', '/base_footprint',
                rospy.Time(0))

        except tf.LookupException:
            return None

        xform2d = transform2d.transform2d_from_ros_transform(ros_xform)

        return xform2d

    # Used to ramp up speed slowly from zero
    def envelope(self, error, tol, gain,
                 min_speed, ramp_time, ramp_speed):

        time = (rospy.get_rostime() - self.start_time).to_sec()
        command = error * gain
        command = numpy.sign(command) * max(abs(command), min_speed)
        effective_time = min(time, ramp_time)
        max_speed = 0.8*(effective_time * ramp_speed/ramp_time)
        command = numpy.clip(command, -max_speed, max_speed)
        done = abs(error) < tol

        return command, done

    # Just calls function above with angular constants
    def angular_envelope(self, error):
        return self.envelope(error, ANGULAR_TOL, ANGULAR_GAIN,
                             ANGULAR_MIN_SPEED,
                             ANGULAR_RAMP_TIME, ANGULAR_RAMP_SPEED)

    # Just calls function above with linear constants
    def linear_envelope(self, error):
        return self.envelope(error, LINEAR_TOL, LINEAR_GAIN,
                             LINEAR_MIN_SPEED,
                             LINEAR_RAMP_TIME, LINEAR_RAMP_SPEED)
    
    # Called 100 times per second to control the robot.
    def control_callback(self, timer_event=None):

        # Velocity we will command (modifed below)
        cmd_vel = Twist()

        time_since_stop = (rospy.get_rostime() - self.stop_time).to_sec()

        if self.should_stop or time_since_stop < 1.0:
            self.cmd_vel_pub.publish(cmd_vel)
            return

        # Flag for finished with state
        done = False

        # Get current pose
        cur_pose = self.get_current_pose()

        # Try to get relative pose
        if cur_pose is not None:
            if self.start_pose is None:
                self.start_pose = cur_pose.copy()
            rel_pose = self.start_pose.inverse() * cur_pose

        # Dispatch on state:
        if self.state == 'initializing':

            # Go from initializing to idle once TF is ready and we
            # have our first laser scan.
            if cur_pose is not None and self.angles is not None:
                done = True

        elif self.state == 'straighten':

            # Register points at +/- 5 degrees, and examine the angle
            # between them.
            
            points = self.points_at_angles([5*DEG, -5*DEG])

            if points is None:

                # No points - missing scan data?
                rospy.logwarn('points was None in straighten state!')

            else:

                # Subtract points
                delta = points[1]-points[0]

                # Get angle
                angular_error = numpy.arctan2(delta[0], -delta[1])

                # Get command and go to idle state if done
                command, done = self.angular_envelope(angular_error)

                cmd_vel.angular.z = command

        elif self.state == 'turnleft' or self.state == 'turnright':

            if self.state == 'turnleft':
                goal_theta = numpy.pi/2
            else:
                goal_theta = -numpy.pi/2

            angular_error = goal_theta - rel_pose.theta

            if abs(angular_error) < 15*DEG:
                self.reset_state('straighten', False)

            command, done = self.angular_envelope(angular_error)

            cmd_vel.angular.z = command

        elif self.state == 'nudge':

            points = self.points_at_angles([0*DEG])

            if points is None:

                # No points - missing scan data?
                rospy.logwarn('points was None in nudge state!')

            else:

                dist_to_wall = points[0][0]

                rounded_dist = numpy.floor(dist_to_wall/(3*FT))*3*FT + 1.5*FT

                linear_error = dist_to_wall - rounded_dist

                command, done = self.linear_envelope(linear_error)

                cmd_vel.linear.x = command

        elif self.state == 'forward' or self.state == 'backward':
            
            # TODO: write me!
            if self.state == 'forward':
                goal_dist = 0.6 
            else:
                goal_dist = -0.6
        
            linear_error =  goal_dist - rel_pose.x 
            if abs(linear_error) < 0.1:
                self.reset_state('nudge',False)
 
            command, done = self.linear_envelope(linear_error) 
 
            cmd_vel.linear.x = command   
                        
            
        elif self.state != 'idle':

            # Die if invalid state :(
            rospy.logerr('invalid state {}'.format(self.state))
            rospy.signal_shutdown('invalid state')

        # Publish our velocity command
        self.cmd_vel_pub.publish(cmd_vel)

        if done:
            rospy.loginfo('done with state {}'.format(self.labeled_state))
            if self.command_index + 1 >= len(self.maze_commands):
                rospy.loginfo('all done!')
                rospy.signal_shutdown('all done, quitting!')
            else:
                self.command_index += 1
                self.reset_state(self.maze_commands[self.command_index])

    # Running the controller is just rospy.spin()
    def run(self):
        rospy.spin()

# Really skinny main function
if __name__ == '__main__':
    try: 
        c = Controller()
        c.run()
    except rospy.ROSInterruptException:
        pass
