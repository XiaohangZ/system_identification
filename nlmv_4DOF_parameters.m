%% Parameters of 4-DOF maneuvering model

%% Load wave force RAO
load('vessels/rao_nonlinear_4DOF_maneuvering.mat', 'RAO');

rng('shuffle');

%% Enable noisy effects (environmental, sensor, ...)
enable.wind = true;
enable.wave = true; % only effective if wind is also true
enable.ocean_current = false;
enable.sensor_noise = true;

control.mode.OPEN_LOOP = 1;
control.mode.CLOSED_LOOP = 2;
control.mode.ZIGZAG = 3;
control.mode.CIRCLE = 4;
control.mode.PULLOUT = 5;
control.mode.SPIRAL = 6;
control.mode.OPEN_LOOP_ROUTINE = 7;

enable.control_mode = control.mode.OPEN_LOOP;

VSS_open_loop = Simulink.Variant('enable.control_mode==control.mode.OPEN_LOOP');
VSS_closed_loop = Simulink.Variant('enable.control_mode==control.mode.CLOSED_LOOP');
VSS_zigzag = Simulink.Variant('enable.control_mode==control.mode.ZIGZAG');
VSS_circle = Simulink.Variant('enable.control_mode==control.mode.CIRCLE');
VSS_pullout = Simulink.Variant('enable.control_mode==control.mode.PULLOUT');
VSS_spiral = Simulink.Variant('enable.control_mode==control.mode.SPIRAL');
VSS_open_loop_routine = Simulink.Variant('enable.control_mode==control.mode.OPEN_LOOP_ROUTINE');

%% Initial ship state
% initial control: rudder angle, rpm
init.rudder_angle_left = 0;
init.rudder_angle_right = 0;
init.rpm_left = 915;
init.rpm_right = init.rpm_left;

% initial velocities and position
init.v0 = [5, 0, 0, 0, 0, 0]';
init.eta0 = [0, 0, 0, 0, 0, 0]';

%% Environmental constants
seawater_density = 1025; % kg/m^3

%% Propeller
% propeller command
prop.max_rpm = 2000;
prop.max_rate = 10;
prop.tc = 2;
% propeller parameters
prop.wake_factor = 0.25;
prop.diameter = 0.75; % [m]
% T_VV, T_Vn, T_nn
prop.ktp = [-0.1060 -0.3246 0.4594]; % thrust coefficients
prop.kqp = [-0.0186 -0.0399 0.0680]; % torque coefficients
prop.reverse_reduction = 0.8;

%% Rudder
% rudder command
rudder.max_angle = (pi*30)/180; % TODO: different max values for different speeds
rudder.max_rate = (pi*12)/180;
rudder.tc = 1;

% rudder propeller parameters
% assumption: COG approx CO, L_oa=55m, B=8.6m, height over water=7.5m
rp.lx = -25; % [m]
rp.ly = 4; % [m] (right propeller = ly, left propeller = -ly)
rp.lz = 4; % [m] 

%% Wind: Ship coefficients
wind.L = 52.5; % [m] length overall
wind.B = 8.6; % [m] beam
wind.A_L = 287.5; % [m^2] lateral projected area
wind.A_F = 71.0; % [m^2] frontal projected area
wind.A_SS = 81.25; % [m^2] lateral projected area of superstructure
wind.S = 74.5; % [m] length of perimeter of lateral projection of model excluding waterline and slender bodies such as masts and ventilators
wind.C = 24.0; % [m] distance from bow to centroid of lateral projected area
wind.M = 2; % number of groups of masts seen in lateral projection

%% Wind: Generator settings (NORSOK wind spectrum)
windg.speed_angle_variation = true;
windg.wind_gust = false;

windg.mean_velocity = 5; % [m/s]
windg.mean_angle = -pi; % [rad]
windg.height = 10; % [m]
windg.sample_time = 5; % [s]
windg.speed_tc = 10; % [s]
windg.angle_tc = 10; % [s]
windg.speed_sat = 1; % [m/s]
windg.angle_sat = pi/4; % [rad]
windg.speed_noise_power = 1;
windg.angle_noise_power = 0.1;
windg.sea_surface_drag = 0.0026; % constant for Harris wind spectrum
windg.scaling_length = 1800; % ""
windg.min_gust_frequency = 1E-4; % [Hz]
windg.max_gust_frequency = 0.1; % [Hz]

windg.speed_seed = randi([0 5000]);
windg.angle_seed = randi([0 5000]);
windg.gust_seed = randi([0 5000]);

%% Waves
% Wave spectrum JONSWAP (wind-induced)
wave.random_frequencies = true;
wave.random_direction = true;

wave.fetch = 100000;
wave.spreading_factor = 2; % values (1,2,3,4,5)
wave.number_frequencies = 20; % >= 20
wave.number_directions = 10; % >= 10
wave.component_energy_limit = 0.005; % interval [0, 1] or as number of waves
wave.frequency_cutoff = 3; % interval [1.5, 5]
wave.direction_cutoff = 0; % interval [0, 3pi/8]
wave.seed = randi([0 5000]);

%% Ocean current
current.speed_variation = true;

current.mean_speed = 1; % [m/s]
current.sideslip_angle = 66*pi/180; % [rad]
current.speed_noise_power = 20;
current.sample_time = 5;
current.speed_sat = 3;
current.speed_tc = 20;

%% Sensor
% Selects measurable system state and applies band-limited Gaussian white noise

% Sample times
sensor.sample_time = 0.1; % [s]
sensor.noise_time = 0.1; % [s]

% Noise powers
sensor.noise_power.position = [1.5 1.5 0 0 0 0].^2;
sensor.noise_power.velocity_over_ground = [0.1 0.1 0 0 0 0].^2; 
sensor.noise_power.velocity_through_water =  [0.5 0.5 0 0 0 0].^2;
sensor.noise_power.wind_direction = (1*pi/180)^2;
sensor.noise_power.wind_speed = (0.1)^2;
sensor.noise_power.propeller_rev = (4)^2; % 0.2% of max rpm
sensor.noise_power.rudder_angle = (0.06*pi/180)^2; % 0.2% of max angle

% Offset
sensor.offset.velocity_through_water = [1 1 0 0 0 0];

% Random seed
sensor.seed.position = randi([0 5000]);
sensor.seed.velocity_over_ground = randi([0 5000]);
sensor.seed.velocity_through_water = randi([0 5000]);
sensor.seed.wind_direction = randi([0 5000]);
sensor.seed.wind_speed = randi([0 5000]);
sensor.seed.propeller_rev.left = randi([0 5000]);
sensor.seed.propeller_rev.right = randi([0 5000]);
sensor.seed.rudder_angle.left = randi([0 5000]);
sensor.seed.rudder_angle.right = randi([0 5000]);

%% Control: Waypoint Following
control.wp.waypoints = [500 0; 1000 0; 1000 500; 1000 750; 1000 1000; 800, 1200; 300, 1400; 0, 1400]; % [m] dim: N x 2, N waypoints
control.wp.speed = 6 * ones(1, length(control.wp.waypoints)); % [m/s] dim: N, desired speed until waypoint

% [m], dim: N,  radius around waypoint which is considered as close enough to waypoint, generally set to 2*L_oa
control.wp.radius = waypoint_radius(init.eta0(1:2)', control.wp.waypoints, wind.L);
% it may be desirable to set the last radius of the last waypoint to a low value
control.wp.last_wp_radius = 150;

% PID controller for rudder
% Controller for standard straight-lines (c2)
control.wp.rudder_P = 0.5;
control.wp.rudder_I = 0.0001;
control.wp.rudder_D = 0;
control.wp.rudder_N = 0;

% PID controller for propeller
control.wp.propeller_P = 0;
control.wp.propeller_I = 0;
control.wp.propeller_D = 0;
control.wp.propeller_N = 0;
control.wp.propeller_B = 0;

%% Control: Markov Model
control.markov.op_seed = randi([0 5000]);
control.markov.speed_seed = randi([0 5000]);

% Cumulative probabilities for operation mode selection
% Forward, Adjust, Turn, Circle, Curve=1.0
control.markov.prob_initial = [0.96 0.97 0.98 0.99];
% Adjust, Turn, Circle, Curve=1.0
control.markov.prob_forward = [0.4 0.6 0.8];

control.markov.forward_duration = [30,360]; % [s]
control.markov.adjust_duration = [15,30]; % "
control.markov.hardturn_duration = [45,90]; % "
control.markov.circle_duration = [120,240]; % "
control.markov.longcurve_duration = [120, 240]; % "

control.markov.forward_angle = 15*pi/180; % [rad]
control.markov.adjust_angle = 30*pi/180; % "
control.markov.hardturn_angle = 30*pi/180; % "
control.markov.circle_angle = 30*pi/180; % "
control.markov.longcurve_angle = 0.5*30*pi/180; % "

control.markov.slow_duration = [60, 480]; % [s]
control.markov.medium_duration = [60, 480]; % "
control.markov.fast_duration = [60, 480]; % "

control.markov.slow_speed_range = [2, 4]; % [m/s]
control.markov.medium_speed_range = [4, 6]; % "
control.markov.fast_speed_range = [6, 9]; % "

control.markov.rpm_rate_limit = 50; % [RPM]
control.markov.angle_rate_limit = 6*pi/180; % [rad]

%% Control: Routine Operation Markov
control.routine.op_seed = randi([0 5000]);
control.routine.speed_seed = randi([0 5000]);

% Cumulative probabilities for operation mode selection
% Forward, Adjust, Curve=1.0
control.routine.prob_initial = [0.98 0.99];
% Adjust, Turn, Circle, Curve=1.0
control.routine.prob_forward = 0.8;

control.routine.forward_duration = [60,360]; % [s]
control.routine.adjust_duration = [5,20]; % "
control.routine.longcurve_duration = [60, 120]; % "

control.routine.slow_duration = [60, 480]; % [s]
control.routine.medium_duration = [60, 480]; % "
control.routine.fast_duration = [60, 480]; % "
control.routine.slow_speed_range = [1, 3]; % [m/s]
control.routine.medium_speed_range = [3, 5]; % "
control.routine.fast_speed_range = [5, 7]; % "

control.routine.rpm_rate_limit = 50; % [RPM]
control.routine.angle_rate_limit = 3*pi/180; % [rad]

%% Control: ZigZag
control.zigzag.start_time = 10;
control.zigzag.max_rudder_angle = 20*pi/180;
control.zigzag.active_right = true;
control.zigzag.active_left = true;

%% Control: Pullout
control.pullout.start_time = 10; % [s]
control.pullout.return_to_zero = 100; % [s]
control.pullout.rudder_angle = -20*pi/180; % [rad]
control.pullout.active_left = true;
control.pullout.active_right = true;

%% Control: Turning circle
control.turning_circle.rudder_angle = -15*pi/180; % [rad]
control.turning_circle.start_time = 180; % [s]
control.turning_circle.active_right = true;
control.turning_circle.active_left = true;

%% Control: Dieudonne's spiral
control.spiral.start_angle_sign = -1; % -1 or 1 for negative or positive angle
control.spiral.active_right = true;
control.spiral.active_left = true;

control.spiral.start_time = 20; % [s]
control.spiral.step_duration = 100; % [s]
control.spiral.max_angle = 25*pi/180; % [rad]
control.spiral.angle_change = 5*pi/180; % [rad]

%% Isherwood (1972) wind
% copy from Isherwood72.m (GNC)
isherwood.rho_a = 1.224;              % density of air at 20 C

% relative wind directions for CX, CY and CN data
isherwood.gamma_r = (pi/180)*(0:10:180)';   % rad

% CX_data = [A0	A1	A2	A3	A4	A5	A6]
isherwood.CX_data= [...  
 	2.152	-5.00	0.243	-0.164	0	    0       0	
 	1.714	-3.33	0.145	-0.121	0 	    0	    0	
 	1.818	-3.97	0.211	-0.143	0	    0	    0.033	
 	1.965	-4.81	0.243	-0.154	0	    0   	0.041	
 	2.333	-5.99	0.247	-0.190	0    	0	    0.042
 	1.726	-6.54	0.189	-0.173	0.348	0	    0.048	
 	0.913	-4.68	0	    -0.104	0.482	0	    0.052	
 	0.457	-2.88	0	    -0.068	0.346	0	    0.043	
 	0.341	-0.91	0	    -0.031	0	    0    	0.032	
 	0.355	0	    0	    0   	-0.247	0	    0.018	
 	0.601	0	    0	    0	    -0.372	0	    -0.020
 	0.651	1.29	0	    0	    -0.582	0	    -0.031	
 	0.564	2.54	0	    0	    -0.748	0	    -0.024	
 	-0.142	3.58	0	    0.047	-0.700	0	    -0.028	
 	-0.677	3.64	0	    0.069	-0.529	0	    -0.032	
 	-0.723	3.14	0	    0.064	-0.475	0	    -0.032	
	-2.148	2.56	0	    0.081	0	    1.27	-0.027	
	-2.707	3.97	-0.175	0.126	0	    1.81	0	
 	-2.529	3.76	-0.174	0.128	0    	1.55	0	      ];

% CY_data = [B0	B1	B2	B3	B4	B5	B6]
isherwood.CY_data = [...
    0       0       0       0       0       0       0       
 	0.096	0.22	0	    0	    0   	0       0   	
 	0.176	0.71	0	    0	    0	    0	    0   	
 	0.225	1.38	0	    0.023	0	    -0.29	0   	
 	0.329	1.82	0	    0.043   0   	-0.59	0   	
 	1.164	1.26	0.121	0	    -0.242	-0.95	0   	
 	1.163	0.96	0.101	0	    -0.177	-0.88	0   	
 	0.916	0.53	0.069	0	    0   	-0.65	0   	
 	0.844	0.55	0.082	0	    0   	-0.54	0   	
 	0.889	0	    0.138	0	    0   	-0.66	0   
 	0.799	0	    0.155	0	    0    	-0.55	0   	
 	0.797	0	    0.151	0	    0	    -0.55	0   	
 	0.996	0	    0.184	0	    -0.212	-0.66	0.34	
 	1.014	0	    0.191	0	    -0.280	-0.69	0.44	
 	0.784	0	    0.166	0	    -0.209	-0.53	0.38	
 	0.536	0	    0.176	-0.029	-0.163	0	    0.27	
 	0.251	0	    0.106	-0.022	0	    0	    0	    
 	0.125	0	    0.046	-0.012	0	    0	    0	    
    0       0       0       0       0       0       0     ];

% CN_data = [C0	C1	C2	C3	C4	C5]
isherwood.CN_data = [...
    0       0       0       0       0       0       
 	0.0596	0.061	0	    0	    0	    -0.074	
 	0.1106	0.204	0	    0	    0	    -0.170	
 	0.2258	0.245	0   	0	    0	    -0.380	
 	0.2017	0.457	0	    0.0067	0	    -0.472	
 	0.1759	0.573	0	    0.0118	0	    -0.523	
 	0.1925	0.480	0	    0.0115	0	    -0.546	
 	0.2133	0.315	0	    0.0081	0	     -0.526	
 	0.1827	0.254	0	    0.0053	0	    -0.443	
 	0.2627	0	    0	    0	    0	    -0.508	
 	0.2102  0	    -0.0195	0	    0.0335	-0.492	
 	0.1567	0	    -0.0258	0	    0.0497	-0.457
 	0.0801	0	    -0.0311	0	    0.0740	-0.396	
 	-0.0189	0	    -0.0488	0.0101	0.1128	-0.420	
 	0.0256	0	    -0.0422	0.0100	0.0889	-0.463
 	0.0552	0	    -0.0381	0.0109	0.0689	-0.476
 	0.0881	0	    -0.0306	0.0091	0.0366	-0.415
 	0.0851	0	    -0.0122	0.0025	0   	-0.220	
    0       0       0       0       0       0       ];
