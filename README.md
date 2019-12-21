Requirements to install for Genetic Algorithm:
	gym
	py4j
	port_for  #Although I have replace the library with import socket instead, because port_for was not working properly
	opencv-python

	then running pip install -e . within the FightingICEV4.40 folder

To run our AI, you have to navigate into FightingICEV4.40 sub-directory:

	In both BeginnerAI.py and BegInterAI.py, you have to change the line for self.env = gym.make("Fightinginc...", java_env_path="(The directory to the FightingICEV4.40)")
	You will have to manually change the path, so it leads directly to the FightingICEV4.40 sub-directory

	For the genetic algorithm implementation:
		1.) Run BeginnerAI.py

	For the DQN reinforcement algorithm, you will have to take extra steps:
		
		1.) Make sure that your system is compatible with CUDA
		2.) Ensure you are the environment is running on python 3.6
		3.) Install keras and tensorflow-gpu
		4.) Download CUDA toolkit (which includes the driver)
			a.) https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/ for more in-depth details
			b.) at the bottom of https://www.tensorflow.org/install/gpu are described extra steps if it is not working
		5.) Reset computer
		6.) Run BegIterAI.py in the FightingICEV4.40 sub-directory
		7.) If tensorflow-gpu can't locate cudart64_100.dll (only if you get an error)
			i.) Navigate to CUDAs bin directory (C:/Program Files/ NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin
		       ii.) Drag the cudart64_100.dll file from the zip file within the root directory of this project
		8.) Run BegIterAI.py
