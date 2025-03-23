# SCP: Spatial Constraint Planning for Robotic Manipulation in MuJoCo

This project implements Spatial Constraint Planning (SCP) for robotic manipulation tasks within the MuJoCo. It is partially based on the [Rekep](https://rekep-robot.github.io/) and supports testing in [MimicGen](https://mimicgen.github.io/) and [Robosuite](https://robosuite.ai/) environments.
![](scp/images/img.gif)

## Installation Guide

1. **Create a Conda Environment:**
   ```bash
   conda create -n scp python=3.10
   conda activate scp
   ```

2. **Install SCP Package:**
   ```bash
   cd scp/
   pip install -e .
   ```

3. **Install MimicGen:**
   ```bash
   cd ../mimicgen
   pip install -e .
   ```

4. **Install Robosuite:**
   ```bash
   cd ../robosuite
   pip install -e .
   ```

## Running Demos

   ```bash
   cd scp/scp
   python main.py --visualize --use_cached_query
   ```
   - `--visualize`:  Provides visualization of the planned trajectory.
   - `--use_cached_query`: Uses a cached GPT-4o query for faster execution.

## Using Your Own API Key

1. **Configure API Key:**
   - Edit `config.yaml` in the `scp/scp` directory.
   - Set your OpenAI API key in the `API_KEY` field.
   - Optionally, modify `API_BASE`.

2. **Run with Custom Query:**
   ```bash
   python main.py --visualize
   ```

## Running Other Tasks

Use the `--task` argument to specify tasks.

1. **Run a specific task (e.g., Lift):**
   ```bash
   python main.py --task Lift --visualize
   ```

2. **Adding New Task Descriptions:**
   For new tasks, add descriptions to `task_descriptions` in `config.yaml`. Example for "Lift":

   ```yaml
   task_descriptions:
     Lift:
       description: "lift the red cube and lift it straight up vertically"
       env: Lift
   ```
   - `description`: Task description.
   - `env`: Environment name.