# Weave-Agent Architecture

This document provides an overview of the Weave-Agent architecture and its components, focusing on how the agent operates and interacts with external systems.

## Core Components

### WeaveAgentTree

The `WeaveAgentTree` serves as the central controller for all agent nodes. It:
- Maintains the global state of all agent nodes
- Manages their relationships in a hierarchical structure
- Handles the event stream of block generation
- Validates transitions between block types
- Provides methods for running agents and viewing problem state

### WeaveAgentNode

Each `WeaveAgentNode` represents an autonomous agent that can:
- Solve specific tasks
- Spawn child agents (subagents)
- Return results to its parent
- Operate through cognitive cycles called "ticks"

### Tick

A `Tick` represents one complete cognitive cycle of the agent, consisting of:
1. Task reminders and observations
2. Orientation reasoning
3. Action generation and execution
4. Expectation formation
5. Observation inference and updates
6. Evaluation setup and execution
7. Outcome recording

### Block Types

The agent uses various block types for different cognitive operations:

- **OBSERVATION**: Sensory perception of the environment. This is where tools render their state.
- **TASK_REMINDER**: Reminder of current tasks/goals, showing the hierarchical task structure.
- **ORIENTATION**: Reflective reasoning about the situation. Uses a dialogue format between different "postures" (like EXPECTATION, RATIONAL, EMPIRICISM, CONCLUSION) to analyze the current state.
- **ACTION**: Code to execute actions in the environment. Sets up callback functions that will be executed.
- **EXPECTATION**: Predictions about action outcomes, explicitly stating what success and failure look like.
- **OBSERVATION_INFERENCE**: Management of observation callbacks, adding or removing based on what information will be needed.
- **EVALUATION**: Assessment of action results through callback functions that check against expectations.
- **OUTCOME**: Final results of an action/evaluation, summarizing what happened.
- **ERROR**: Error reporting and handling when things go wrong.
- **DEBUG**: Error diagnosis and troubleshooting to recover from failures.

## Tools System

Weave-Agent interacts with external systems through a set of tools. Each tool follows a common pattern:

1. Registers with the agent system
2. Adds an observation view that renders the tool's state
3. Provides methods for the agent to interact with external systems
4. Has a close method to clean up resources

### Available Tools

#### WeaveEditor

A text editor tool that allows the agent to manipulate files:
- Open, read, modify, and save files
- Navigate through files (up, down, find text)
- Make changes through standard editing operations or unified diff format

```python
editor = WeaveEditor(agent, filepath)
editor.edit(start_line, end_line, new_text)
editor.append(text)
editor.find(regex)
```

#### WeaveDiscordClient

Enables the agent to interact with Discord channels:
- Send messages to Discord channels
- Reply to specific messages
- React to messages with emojis
- Retrieve message history

```python
discord_client = WeaveDiscordClient(agent, token, channel_id)
discord_client.send_message(content)
discord_client.reply_to_message(message_id, content)
discord_client.get_messages()
```

#### WeaveNethack

Allows the agent to play the game Nethack:
- Send keyboard commands to the game
- Track game state and move history
- Visualize the game screen

```python
nethack = WeaveNethack(agent)
nethack.send_keys(command)
```

#### ATSPIDesktopTurtle

Provides desktop automation capabilities:
- Control mouse movement with "turtle-like" commands
- Type text and perform keyboard combinations
- Identify and interact with UI elements
- Scan the screen for interactive elements

```python
desktop = ATSPIDesktopTurtle()
desktop.goto(x, y)
desktop.input_string(text)
desktop.input_key_combination(keys)
```

## Agent Lifecycle

1. **Initialization**: The agent tree is created with a root agent
2. **Bootstrap**: Initial setup code is executed to define tasks and tools
3. **Task Decomposition**: Problems are broken down into subagents with specific tasks
4. **Cognitive Cycles**: Each agent operates through ticks with the phases listed above
5. **Task Evaluation**: Actions are evaluated against expectations
6. **Result Propagation**: Results are returned up the agent tree

## Bootstrap Process

Bootstrap files are used to initialize the agent with specific tasks and tools. They follow a standard format with blocks that demonstrate the agent's cognitive process:

```python
#startblock type: orientation
"""
WEAVER [P: EXPECTATION], I need to create a Discord bot that will interact with users...
WEAVER [P: CLARIFICATION], How do I set up the Discord bot?
WEAVER [P: EXPOSITION], You can set up the Discord bot using the provided Discord tool...
WEAVER [P: RATIONAL], The bot should be able to send and receive messages...
WEAVER [P: CONCLUSION], I will create the main subagent to handle the interaction...
"""
#endblock

#startblock type: action
def create_main_agent_and_connect_discord(subagent):
    # Set up main agent with tasks and schema
    schema = {"dismissed": "boolean"}
    main_agent = agent.subagent("main", None, "Interact with users until dismissed", schema, args.budget)
    main_agent.task.add_evaluation("Check if dismissed", check_dismissed)
    
    # Initialize tools
    client = WeaveDiscordClient(main_agent, token, channel_id)
    
    return True

self.add_action("Create main agent and connect to Discord", create_main_agent_and_connect_discord)
#endblock

#startblock type: expectation
"""
The action should succeed if the Discord bot connects to the channel...
The action should fail if the bot does not connect to the channel...
"""
#endblock

#startblock type: evaluation
def check_bot_connected(subagent):
    """Check if the bot has connected to the Discord channel."""
    # Evaluation logic here
    return True

self.add_evaluation("Check bot connected", check_bot_connected)
#endblock
```

Each bootstrap file:
1. Sets up the main agent with its tasks and schema
2. Initializes the necessary tools (Discord, Nethack, Editor, etc.)
3. Defines evaluation criteria for task completion
4. Provides initial expectations and actions

## Advantages Over Traditional ReAct Agents

1. **Improved Perception**: Uses observation callbacks to better perceive the environment, similar to Cradle's WIMP (Window, Icon, Menu, Pointer) interface but without relying on a visual language model.

2. **Expectation-driven Evaluation**: Writes down explicit expectations and uses evaluation callbacks to check that expected changes actually occurred, improving reliability.

3. **In-distribution Format**: Uses Python as the trace format, keeping the agent in-distribution with code generation tasks rather than using JSON tool calling.

4. **Hierarchical Problem Solving**: Implements a tree-based agent structure for complex task decomposition, allowing recursion and delegation.

5. **Contextual Memory**: Maintains context through a searchable index of past blocks, helping the agent recall and search through its history.

6. **Simulated Planning**: Includes planning capabilities where the agent can simulate actions and their outcomes before executing them.

7. **Morpheus Discussion Format**: Uses a structured dialogue format in orientation blocks where different cognitive postures discuss the problem, similar to a committee of experts.

8. **Structured Error Recovery**: Implements specific DEBUG blocks for error diagnosis and recovery, allowing the agent to reason about what went wrong.