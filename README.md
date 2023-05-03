# Learning Japanese With Waifu

## Description

A similar app to Duolingo but adding more features like Waifu voice and also adding a writing component for users to practice writing simple anime characters. This repository houses the Flask server of this repo.

## Installation

To run this Flask server, you need to install Python 3.8 and above and pip, the Python package manager, as well as the following dependencies:

- Flask
- Flask-Cors
- TensorFlow
- Pillow
- NumPy
- OpenCV

Here are the steps to install these dependencies:

1. Install Python 3.8 and above: 
   - For Windows, go to https://www.python.org/downloads/windows/ and download the latest version of Python 3.8. Follow the installation instructions.
   - For macOS, go to https://www.python.org/downloads/mac-osx/ and download the latest version of Python 3.8. Follow the installation instructions.
   - For Linux, open a terminal and run the following command:
   
     ```sh
     sudo apt-get install python3
     ```
   
2. Install pip:
   - For Windows, download get-pip.py from https://bootstrap.pypa.io/get-pip.py and run the following command:
   
     ```sh
     python get-pip.py
     ```
     
   - For macOS and Linux, open a terminal and run the following command:
   
     ```sh
     sudo apt-get install python3-pip
     ```
3. Create a virtual environment:
   - Open a terminal and Run the following command to create a virtual environment:
     ```sh
     python -m venv <INSERT VIRTUAL EVIRONMENT NAME>
     ```
     You can give the virtual environment any name you desire
     
   - This will create a new directory called "env" that contains the virtual environment for your project.
   
4. Activate the virtual environment:
   - Run the following command to activate the virtual environment:
   
     ```sh
     source <INSERT VIRTUAL ENVIRONMENT NAME>/bin/activate
     ```
     
   - You should now see the name of your virtual environment in your terminal prompt.
   - Add it to .gitignore file
   
5. Install the required dependencies:
   - Open a terminal and run the following command:
   
     ```sh
     pip install Flask Flask-Cors tensorflow Pillow numpy opencv-python-headless
     ```

This will install Flask, Flask-Cors, TensorFlow, Pillow, NumPy, and OpenCV. 

6. Running Flask Server:
   - In your termial run the followig command:
   ```sh
   python server.py
   ```
   - This will start the server and you should be able to access it at http://localhost:5000/.

## Repository Links
1. <a href="https://github.com/LoyChaiEe/capstone_project_frontend">Frontend</a>
2. <a href="https://github.com/LoyChaiEe/capstone_project_backend">Backend</a>
3. <a href="https://github.com/LoyChaiEe/capstone_flask">Flask</a>
