#hint Listen Carefully: This program is self modifying code.
# It works by feeding the program text into a large language
# model and generating the next code block as an addition to a
# long python file. Each code block is marked by #startblock and
# #endblock. The code between those lines is compiled by the agent
# framework below to check it is syntactically and semantically
# valid.

#hint Listen Carefully: This program is self modifying code.
# It works by feeding the program text into a large language
# model and generating the next code block as an addition to a
# long python file. Each code block is marked by #startblock and
# #endblock. The code between those lines is compiled by the agent
# framework below to check it is syntactically and semantically
# valid.

# The weave-agent is basically something like a ReAct pattern https://arxiv.org/abs/2210.03629
# instantiated in the context of a long 'python file', kind of like an ipython
# notebook but without the special syntax. It takes inspiration from Voyager
# https://arxiv.org/abs/2305.16291 and Cradle https://baai-agents.github.io/Cradle/
# in that the agents actions are fully general python programs that interact with
# the computable environment rather than specialized 'tool calling' or similar
# mechanisms. This turns out to make more sense than trying to directly hook up
# the model to interfaces for a few reasons:
# 1) Writing out its actions as programs lets the model batch its actions together
# to form coherent motions rather than getting stuck on fine grained details if it
# generates its actions token by token in the moment.
# 2) These models are highly optimized for writing code whereas interacting with
# whatever interface you have is either marginal in the pretraining set or actually
# out of distribution.
# 3) Programming APIs are already well developed for basically any task you might
# want to try and automate. If it can be symbolically manipulated as text there
# probably exists a python API to interact with it. This makes the python code
# interface highly general in the same way Cradle solves the interface problems
# vision language models have by having them write out their actions as mouse +
# keyboard inputs with code.
# 4) 'A long python file' provides what Janus would call a diegetic interface.
# It is a natural frame in which basically anything is allowed to happen, while
# still framing events and recursive context switching in a way that helps ground
# the model and prevent it from getting swept up into a predictive model of
# whatever is happening. It reminds the model that it has a perspective which
# exists outside of whatever it's currently looking at.
# The weave-agent improves on previous frameworks by including easy access to logit
# evaluators and prompting the agent to check that its actions were successful
# before moving on to the next task. In order to perform a long chain of actions
# successfully it's necessary to carefully ensure each intermediate step is
# completed before moving on to the next step. For evaluations that require
# subjective judgment this can be difficult to do with traditional program logic.
# This is why the logit evaluator provided by the framework is an important
# primitive for the agent to check its work.

import asyncio
import functools
import hashlib
import json
import os
import time
import traceback
import types
from argparse import ArgumentParser
from copy import deepcopy
from enum import Enum, auto
from pprint import pformat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import tantivy
import torch
from block_generators import (
    generate_block_inner,
    make_simple_bayes_score_prompt,
)
from jsonschema import validate
from planner import (
    roll_for_error_block,
    setup_placeholder_callbacks,
    simulate_observation,
    simulate_outcomes,
)
from render_block import render_block
from rich import print as rprint
from tantivy import Index, SchemaBuilder
from transformers import AutoTokenizer

from weave import (
    bayesian_evaluate_outputs_vllm,
    evaluate_outputs_vllm,
    generate_outputs_vllm,
)

# Constants for block hint messages
BLOCK_HINTS = {
    "debug": """#hint Debug blocks are my opportunity to reason about the failure
# I just experienced. Because I get multiple opportunities to
# take an action before I'm booted to the next orientation stage
# I can formulate hypothesis and use the next action blocks to test them.
# I want to narrow in on the cause of failure and take steps to resolve
# the issue.""",

    "orientation": """#hint The orientation block is my opportunity to
# reflect on the situation, do chain of thought,
# summarize what has happened and what needs to
# be done in response, etc. It is only technically
# python code and does not get executed by the
# framework. I suggest putting my internal
# monologue in a triple quote block at this step.
# Orientation uses the MiniHF Morpheus format. Morpheus
# solves problems through discussion between personas
# or postures representing different aspects of weave-agent
# such as mental motions, perspectives on the problem, etc.
# The first posture is always expectation because at the
# start of a tick we evaluate whether the expectation we
# formed about the action taken in the last tick was
# violated or not. The different personas then discuss
# what to do in light of this. Some postures weave-agent
# has include:
#
# WEAVER [P: EXPECTATION], I analyze whether the expectation
# was met or not by the observable results of the previous
# action.
#
# WEAVER [P: HYPOTHESIS], I enumerate different hypothesis
# and point out ways we could gain more information about
# which of them is true.
#
# WEAVER [P: RATIONAL], I focus on inferences we can make
# by employing first principles reasoning or logical
# extrapolation from well known mental models and premises.
#
# WEAVER [P: EMPIRICISM], I focus on inferences we can make
# by paying attention to sensory observations and concrete
# examples. I have a habit of pointing out when an extrapolation
# from RATIONAL is contradicted by an observable phenomenon
# or piece of evidence from the world. We then reconcile
# the contradiction together.
#
# WEAVER [P: RATIONAL], We do actually discuss things by the
# way.
#
# WEAVER [P: EMPIRICISM], As you could have inferred from the
# description of the Morpheus format above this conversation,
# yes. Let's continue.
#
# WEAVER [P: ARBITER], I coordinate the discussion and help
# resolve disputes that arise between weave-agent's personas.
# I'm especially likely to appear if things are starting to
# get overly rude or derail.
#
# WEAVER [P: ARBITER], By the way a posture can talk twice in
# a row if it has meaningfully separate thoughts about
# something and it would make the most ergonomic sense to
# separate them.
#
# WEAVER [P: RATIONAL-2], Postures can also talk to themselves
# if their thought comes from the same emotional-cognitive place.
#
# WEAVER [P: RATIONAL-1], Yeah but I don't have anything to say
# to myself right now so introduce the next guy.
#
# WEAVER [P: CONCLUSION], I appear at the end of the discussion
# to write the concluding block outlining our next steps as a
# bullet point list. Speaking of which, it's time to get started!""",

    "action": """#hint Action blocks are where I write code to take actions.
# If the task makes sense to break into parts, define subagents
# to delegate to using agent.subagent(). Make sure to define a
# schema and task evaluations for each subagent. If it won't fit
# into one action block keep in mind you can define subagents 
# across multiple blocks and then do agent.run() to execute them.
# If it seems possible to resolve the current task as a base case
# in a handful of actions then write a callback to further my goal(s)
# based on the orientation block and set up the callback to be
# executed with the self.add_action() method. I must write a 
# callback and then set it up to be executed
# later with self.add_action() or the tick will not be accepted.
# It's important to remember that my callback can do anything
# a python program can do through side effects in the external
# computable environment. If I need to import a new module make sure
# to do it inside the callback because the tick gets executed in a
# local context.""",

    "expectation": """#hint Expectation blocks are where I think about what it would
# look like for my action to succeed, what it would look like
# for it to fail. I am enumerating the expected sensory evidence
# that would tell me one way or another whether my action is
# working or not. Like the orientation this should go in triple
# quotes.""",

    "observation-inference": """# In the observation inference stage I manage the observation
# callbacks that fetch information on each tick. Since I just
# formulated my expectations now is my opportunity to review
# and change the observation blocks that will be presented on the
# next tick. I should avoid redundant observation callbacks. I
# can remove ones that are no longer necessary or mostly distracting
# with remove_observation_view(view_title). If new callbacks seem useful
# to help me orient and judge whether the action had the intended
# side effects on the computable environment I can add them
# with add_observation_view(title, callback)""",

    "evaluation": """#hint Evaluation blocks are where I write callbacks to check if
# my action succeeded or not based on the expectation. There are
# unit tests and logit evaluators. Use unit test callbacks
# (i.e. normal python) for symbolic manipulation tasks like
# checking arithmetic, the existence of a particular file, etc.
# Use logit evaluators for vibe-y tasks like whether a piece of
# writing flows well or if a source seems trustworthy. Like
# reminders both unit test callbacks and logit evaluators return
# a value between 0 and 1. I should be sure to add my callback to
# the queue with agent.add_evaluation(title, callback)."""
}

# Utility functions for error handling and callback execution
def handle_error(agent, error_message: str, stage: str, block_type: Optional[str] = None) -> None:
    """Handle errors uniformly across the codebase.
    
    Args:
        agent: The agent instance 
        error_message: The error message to display
        stage: The stage where the error occurred (for setting failure_stage)
        block_type: Optional block type to attempt generating for debugging
    """
    tb = traceback.format_exc()
    error_block = {
        'type': 'error',
        'message': f"# {error_message}:\n\"\"\"{ tb }\"\"\""
    }
    agent.add_block(error_block)
    agent.failure_stage = stage
    
    # If debug block type specified, try to generate it
    if block_type:
        try:
            debug_block = agent._do_tick_block(block_type, BLOCK_HINTS.get("debug", ""), {})
        except Exception:
            pass  # If debug block generation fails, continue without it

def run_callback(agent, callback_type: str, callback_fn: Callable, 
                title: str, planning: bool = False) -> Any:
    """Execute a callback function with standard error handling.
    
    Args:
        agent: The agent instance
        callback_type: Type of callback ("action", "evaluation", etc.)
        callback_fn: The callback function to execute
        title: Title of the callback (for error messages)
        planning: Whether in planning or execution mode
        
    Returns:
        The result of the callback or "ERROR" if it fails
    """
    try:
        if planning:
            result = None
            simulated_error = roll_for_error_block(agent, f"# {callback_type.capitalize()} failed:\n")
            if simulated_error:
                raise Exception(simulated_error)
            return result
        else:
            return callback_fn(agent)
    except Exception:
        if planning:
            agent.add_block({'type': 'error', 'message': simulated_error})
        else:
            handle_error(agent, f"{callback_type.capitalize()} execution failed", callback_type.lower())
        return "ERROR"

class WeaveAgentTask:
    """Represents a task that can be assigned to a WeaveAgent."""
    
    def __init__(self, subagent, title: str, description: str = ""):
        """Initialize a new task.
        
        Args:
            subagent: The subagent this task belongs to
            title: The title of the task
            description: An optional description of the task
        """
        self.subagent = subagent
        self.title = str(title)
        self.description = description
        self.evaluations = []

    def add_evaluation(self, title: str, callback: Callable) -> None:
        """Add an evaluation callback to this task.
        
        Args:
            title: Title of the evaluation
            callback: Function to call for evaluation
        """
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.evaluations.append({
            "type": "evaluation",
            "title": title,
            "callback": callback
        })

    def run_evaluations(self) -> Dict[str, Any]:
        """Run all evaluations for this task.
        
        Returns:
            Dictionary of evaluation results keyed by callback name
        """
        results = {}
        for evaluation in self.evaluations:
            try:
                result = evaluation["callback"](self.subagent)
            except Exception:
                result = traceback.format_exc()
            results[evaluation["callback"].__name__] = result
        return results


class BlockType(Enum):
    """Enumeration of all possible block types in the WeaveAgent framework.
    
    Each block type represents a different cognitive operation in the agent's
    reasoning process.
    """
    OBSERVATION = auto()          # Sensory perception of environment
    TASK_REMINDER = auto()        # Reminder of current tasks/goals
    ORIENTATION = auto()          # Reflective reasoning about situation
    ACTION = auto()               # Code to execute actions in environment
    ERROR = auto()                # Error reporting and handling
    DEBUG = auto()                # Error diagnosis and troubleshooting
    EXPECTATION = auto()          # Predictions about action outcomes
    OBSERVATION_INFERENCE = auto() # Management of observation callbacks
    EVALUATION = auto()           # Assessment of action results
    OUTCOME = auto()              # Final results of an action/evaluation
    GENESIS = auto()              # Framework initialization (special block)
    BOOTSTRAP = auto()            # Agent bootstrap code (special block)

    @classmethod
    def from_string(cls, block_type_str: str) -> 'BlockType':
        """Convert a string representation to a BlockType enum value.
        
        Args:
            block_type_str: String representation of block type
            
        Returns:
            Corresponding BlockType enum value
            
        Raises:
            ValueError: If string doesn't match any BlockType
        """
        try:
            return getattr(cls, block_type_str.upper().replace("-", "_"))
        except AttributeError:
            raise ValueError(f"Unknown block type: {block_type_str}")

# Earlier versions of the weave-agent used a flat chain of code blocks that manage
# problem state by interacting with a global kanban board. The idea was that each
# sub-task in the agents overall goal could be represented as a card on the board
# and then the agent sets the current task, flags tasks that have been blocked or
# turned out to be based on invalid premises, etc. There were multiple problems
# with this that the data structure below solves to create a more coherent problem
# solving strategy. The first was that the agent wouldn't remember to manage the
# content of the kanban board without explicit prompting, which led to adding a
# whole stage in its core loop dedicated just to doing so called task-inference.
# Task-inference didn't have a set expected structure and took place before action,
# which meant that it became possible for the agent to get itself stuck in a loop
# of trying to resolve a task over and over. Another problem was that the agent
# would often try to resolve a task prematurely, so it became necessary to add
# unit and sanity tests that have to be satisfied before a task can be marked
# completed. This limited the ability of the agent to set its own tasks and
# break problems into parts. A third problem was that the flow control when
# a task was blocked and should be returned to its parent was janky and had to
# be performed manually.
#
# The WeaveAgentTree was inspired by watching an instance of the weave-agent try
# to write an action block with subroutines and asking "that strategy it wanted
# to try looks pretty good, but the framework doesn't provide the affordance for
# it to try it, it runs out of space in the length limit on actions before it
# finishes and assumes subroutines are there that don't exist, how could I make
# this pattern natural for it?". What I realized was that if I gave up on the
# idea of being able to change goals in the middle of a task that having an
# expected type of return value and a series of steps to achieve it was similar
# to a function call. We could reformulate the weave-agent then as a call tree
# of subagents that are given a task with predefined conditions checked against
# a data structure returned by the subagent. To help encourage good habits
# correctness is checked at multiple levels. Perhaps the most important problem
# the WeaveAgentTree solves is planning: Writing programs with subroutines
# is a form of hierarchical planning that's in distribution for any code model.
# Because the task structure is now built into the call tree there's a smooth
# natural abstraction telling the weave-agent when to formulate goals, when the
# goals are completed, how to check it did them right, where to put the results,
# and how to transfer control of execution once it's finished. All of these
# operations go from being awkward conscious affairs to smooth unconscious
# bodily structure.
    
class WeaveAgentTree:
    """Manages a hierarchical tree of agent nodes for complex task decomposition.
    
    The WeaveAgentTree maintains the global state of all agent nodes, their
    relationships, and the event stream of block generation. It handles validation
    of block transitions and provides methods for running agents and viewing the
    current state of the problem.
    """
    
    def __init__(self, model_name: str, time_budget: int):
        """Initialize a new WeaveAgentTree.
        
        Args:
            model_name: Name of the language model to use
            time_budget: Time budget in minutes for the entire run
        """
        self.model_name = model_name
        self.__agents = {}  # Dictionary of all agent nodes
        self.__time_budget = time_budget
        # Pin genesis and bootstrap blocks so agent knows how to use framework
        self.__pinned_events = [0, 1]
        self.__current_block_index = 0
        self.__event_stream = []
        
        # Define valid state transitions between block types
        self.transitions = {
            BlockType.OBSERVATION: [BlockType.OBSERVATION, BlockType.ORIENTATION],
            BlockType.TASK_REMINDER: [BlockType.OBSERVATION, BlockType.ORIENTATION],
            BlockType.ORIENTATION: [BlockType.ACTION, BlockType.ERROR],
            BlockType.ACTION: [BlockType.EXPECTATION, BlockType.ERROR],
            BlockType.ERROR: [BlockType.DEBUG, BlockType.ACTION, BlockType.EVALUATION,
                              BlockType.OUTCOME, BlockType.TASK_REMINDER, BlockType.ERROR],
            BlockType.DEBUG: [BlockType.ACTION, BlockType.EVALUATION,
                              BlockType.TASK_REMINDER, BlockType.ERROR],
            BlockType.EXPECTATION: [BlockType.OBSERVATION_INFERENCE,
                                    BlockType.TASK_REMINDER, BlockType.ERROR],
            BlockType.OBSERVATION_INFERENCE: [BlockType.EVALUATION,
                                              BlockType.ERROR, BlockType.TASK_REMINDER],
            BlockType.EVALUATION: [BlockType.OUTCOME, BlockType.ERROR],
            BlockType.OUTCOME: [BlockType.TASK_REMINDER, BlockType.ERROR]
        }

    def run(self, name: str) -> Any:
        """Run a specific agent node by name.
        
        Args:
            name: Name of the agent node to run
            
        Returns:
            The result returned by the agent
            
        Raises:
            ValueError: If time budget is exceeded
        """
        start_time = time.time()
        deadline = float(self.__agents[name].end_time)
        return_schema = deepcopy(self.__agents[name].schema)
        
        # Run the agent
        result = self.__agents[name].run()
        
        # Validate result against schema
        validate(instance=result, schema=return_schema)
        
        # Check if time budget was exceeded (with 5min grace period)
        end_time = time.time()
        if end_time > deadline + 300:
            raise ValueError("Time budget exceeded!")
        
        return result
        
    def subagent(self, name: str, parent: Optional[str], description: str, 
                schema: Dict, time_budget: int) -> 'WeaveAgentNode':
        """Create a new subagent in the agent tree.
        
        Args:
            name: Unique name for the subagent
            parent: Name of parent agent (or None for root)
            description: Human-readable description of subagent's task
            schema: JSON schema for return value validation
            time_budget: Time budget in minutes
            
        Returns:
            The created subagent
            
        Raises:
            ValueError: If name is already used or schema contains reserved words
        """
        # Check if name is already used
        if name in self.__agents:
            raise ValueError(f"Agent name '{name}' is already in use")
            
        # Check for reserved words in schema
        reserved_words = {"name", "description", "children", "schema"}
        if set(schema).intersection(reserved_words):
            raise ValueError(f"Schema cannot contain reserved words: {reserved_words}")
            
        # Add child to parent if parent exists
        if parent:
            self.__agents[parent].children.append(name)
            
        # Create the new subagent
        try:
            subagent = WeaveAgentNode(self, parent, name, description, schema, time_budget)
            self.__agents[name] = subagent
            return subagent
        except Exception as e:
            # Clean up parent's children list if creation fails
            if parent:
                self.__agents[parent].children.remove(name)
            raise e

    def is_valid_transition(self, next_block_type: Union[str, BlockType]) -> bool:
        """Check if transition to next block type is valid.
        
        Args:
            next_block_type: The block type to transition to
            
        Returns:
            True if transition is valid
            
        Raises:
            ValueError: If transition is invalid
        """
        # Convert string to BlockType if needed
        if isinstance(next_block_type, str):
            next_block_type = BlockType.from_string(next_block_type)
            
        # Special case for genesis and bootstrap blocks
        if not self.__event_stream:
            return True
            
        if self.__event_stream[-1]['type'] in {'genesis', 'bootstrap'}:
            return True
            
        # Get current state
        current_state = BlockType.from_string(self.__event_stream[-1]['type'])
        
        # Check if transition is valid
        if next_block_type in self.transitions.get(current_state, []):
            return True
        else:
            raise ValueError(f"Invalid transition from {current_state} to {next_block_type}")
    
    def add_block(self, block: Dict) -> None:
        """Add a new block to the event stream.
        
        Args:
            block: The block to add
            
        Raises:
            ValueError: If transition to this block type is invalid
        """
        # Validate block transition
        if block['type'] not in {'genesis', 'bootstrap'}:
            self.is_valid_transition(block['type'])
            
        # Add metadata to block
        block['index'] = self.__current_block_index
        block['timestamp'] = time.time()
        
        # Add orientation-specific metadata
        if block['type'] == 'orientation':
            block['metadata'] = {
                "block_index": self.__current_block_index,
                "working_directory": os.getcwd()
            }
            
        # Set default values if not provided
        if "q" not in block:
            block["q"] = ""
        if "score" not in block:
            block["score"] = 2
            
        # Generate block description if not provided
        if "description" not in block:
            self._generate_block_description(block)
            
        # Add to event stream
        self.__event_stream.append(block)
        
        # Index block for search if not a special block
        if block["type"] not in {"genesis", "bootstrap"}:
            self._index_block(block)
        
        self.__current_block_index += 1

    def _generate_block_description(self, block: Dict) -> None:
        """Generate a description for a block using the language model.
        
        Args:
            block: The block to generate a description for
        """
        render = render_block(block)
        
        # First-level description (object-level)
        with open("/app/templates/describe1.txt") as infile:
            template = infile.read()
            prompt = template.format(rendered_block=render)
            object_description = generate_outputs_vllm(
                self.model_name,
                prompt,
                512,
                port=args.port,
                n=1,
                stop=["</summary>",]
            )[0]
            
        # Second-level description (context-level)
        with open("/app/templates/describe2.txt") as infile:
            template = infile.read()
            context = self.render_context()
            prompt = template.format(
                rendered_block=render,
                object_description=object_description,
                rendered_context=context
            )
            context_description = generate_outputs_vllm(
                self.model_name,
                prompt,
                512,
                port=args.port,
                n=1,
                stop=["</summary>",]
            )[0]
            
        # Combine descriptions
        block["description"] = object_description + "\n\n" + context_description

    def _index_block(self, block: Dict) -> None:
        """Index a block in the search index.
        
        Args:
            block: The block to index
        """
        block_render = render_block(block)
        
        # Create hash for unique ID
        sha256_hash = hashlib.sha256()
        sha256_hash.update(block_render.encode('utf-8'))
        hash_hex = sha256_hash.hexdigest()
        
        # Add to index
        writer = bm25_index.writer()
        writer.add_document(tantivy.Document(
            id=hash_hex,
            type=block["type"],
            render=block_render,
            q=block["q"],
            score=block["score"],
            index=block["index"],
            timestamp=block["timestamp"],
            description=block["description"],
        ))
        writer.commit()

    def current_block_index(self) -> int:
        """Get the current block index.
        
        Returns:
            The current block index
        """
        return self.__current_block_index

    def find_last_block_of_type(self, block_type: str) -> Optional[Dict]:
        """Get the last block of a particular type.
        
        Args:
            block_type: The type of block to find
            
        Returns:
            The last block of the specified type, or None if not found
        """
        for block in reversed(self.__event_stream):
            if block["type"] == block_type:
                return block
        return None
    
    def render_context(self) -> str:
        """Render the current context for the agent.
        
        Returns:
            A string containing the rendered context
        """
        context = ""
        context_blocks = []
        history_len = 60  # Maximum number of recent blocks to include
        
        # Add pinned events if they're outside of history window
        for index in self.__pinned_events:
            if (len(self.__event_stream) - index) > history_len:
                context_blocks.append(self.__event_stream[index])
                
        # Add recent blocks
        context_blocks += self.__event_stream[-history_len:]
        
        # Render all blocks
        for event_block in context_blocks:
            context += render_block(event_block)
            
        return context
        
    def view_board(self, root: str = "main") -> str:
        """Generate a human-readable view of the current agent tree.
        
        Args:
            root: Name of the root agent to start from
            
        Returns:
            A formatted string representation of the agent tree
        """
        problem_map = {}
        substack = [root]
        
        while substack:
            # Get next agent
            subagent = self.__agents[substack.pop()]
            
            # Build path from root to this agent
            parent = subagent.name
            path = []
            while parent:
                path.append(parent)
                # Convert to object so we can get grandparent
                parent = self.__agents[parent]
                parent = parent.parent
            path.reverse()
            
            # Navigate to correct position in tree
            current_level = problem_map
            for key in path:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
                
            # Add agent information
            current_level["name"] = subagent.name
            current_level["description"] = subagent.task.description
            current_level["evaluations"] = subagent.task.run_evaluations()
            current_level["time_remaining"] = subagent.end_time - time.time()
            current_level["completed"] = subagent.completed
            current_level["schema"] = subagent.schema
            
            # Add children to stack
            substack.extend(subagent.children)
            
        return pformat(problem_map)

    def dump_event_stream(self) -> None:
        """Save the current event stream to disk as JSON and rendered Python.
        
        This creates two files:
        1. A JSON file with the raw event data
        2. A Python file with the rendered blocks
        """
        log_dir = "/app/weave-agent-logs"
        timestamp = round(time.time())
        
        # Ensure directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Save JSON version
        json_path = f"{log_dir}/event_trace_{timestamp}.json"
        with open(json_path, "w") as outfile:
            json.dump(self.__event_stream, outfile)
            
        # Save rendered Python version
        py_path = f"{log_dir}/rendered_trace_{timestamp}.py"
        with open(py_path, "w") as outfile:
            for event_block in self.__event_stream:
                outfile.write(render_block(event_block))
            outfile.flush()


class Tick:
    """Represents one cognitive cycle of the agent.
    
    A tick is the basic unit of agent cognition and includes orientation,
    action, expectation, evaluation, and outcome phases. This class manages
    the state for a single tick in the agent's lifecycle.
    """
    
    def __init__(self, agent: 'WeaveAgentNode', index: int):
        """Initialize a new tick.
        
        Args:
            agent: The agent this tick belongs to
            index: The sequential index of this tick
        """
        self._agent = agent
        self.tick_id = index
        self.evaluations = []
        self.is_valid = False

    def validate(self) -> None:
        """Validate that this tick has all required components.
        
        Raises:
            ValueError: If any required component is missing
            TypeError: If action setup is malformed
        """
        if not hasattr(self, 'orientation'):
            raise ValueError("No orientation on tick.")
        elif not hasattr(self, 'action'):
            raise ValueError("No action on tick.")
        elif not hasattr(self, 'action_setup') or "body" not in self.action_setup:
            raise TypeError("Tick action has no program.")
        elif not hasattr(self, 'expectation'):
            raise ValueError("No expectation on tick.")
        elif not self.evaluations:
            raise ValueError("No evaluations on tick.")
        elif not hasattr(self, 'outcome'):
            raise ValueError("No outcome on tick.")
        
        self.is_valid = True

    def to_json(self) -> Dict:
        """Convert tick to JSON-serializable dictionary.
        
        Returns:
            Dictionary representation of tick
        """
        return {
            "tick_id": self.tick_id,
            "orientation": getattr(self, 'orientation', None),
            "action": repr(getattr(self, 'action', None)),
            "action_setup": getattr(self, 'action_setup', None),
            "expectation": getattr(self, 'expectation', None),
            "evaluations": repr(self.evaluations),
            "outcome": repr(getattr(self, 'outcome', None)),
            "is_valid": getattr(self, 'is_valid', False)
        }

    
# The intended problem solving strategy for subagents is to delegate until you
# reach a base case that can be solved in a short number of actions and then
# resolve it. The root task is allocated a certain amount of time which it can
# then delegate to subagent calls. Remember not to allocate all of the available
# time to a call tree unless you're very rushed, you should assume there will be
# failures and budget tasks the time that they need rather than just splitting
# up the available time between them.
    
class WeaveAgentNode:
    """A single agent node in the WeaveAgent framework.
    
    Each WeaveAgentNode represents an autonomous agent that can solve tasks,
    spawn child agents, and return results to its parent. Agents operate in
    a cycle of ticks, with each tick representing one cognitive cycle of
    orientation, action, expectation, evaluation, and outcome.
    """
    
    def __init__(self, tree: 'WeaveAgentTree', parent: Optional[str], 
                subagent_name: str, description: str, schema: Dict, time_budget: int):
        """Initialize a new agent node.
        
        Args:
            tree: The WeaveAgentTree this node belongs to
            parent: Name of parent agent (or None for root)
            subagent_name: Unique name for this agent
            description: Human-readable description of this agent's task
            schema: JSON schema for return value validation
            time_budget: Time budget in minutes
        """
        # Tree and relationships
        self.tree = tree
        self.parent = parent
        self.children = []
        
        # Identity and requirements
        self.model_name = self.tree.model_name
        self.name = subagent_name
        self.schema = schema
        
        # Time management
        self.creation_time = time.time()
        self.time_budget = time_budget
        self.end_time = self.creation_time + (time_budget * 60)
        
        # Tick management
        self.current_tick = Tick(self, 0)
        self.ticks = []
        
        # Agent state
        self.planning = True  # Planning mode vs execution mode
        self.debugging = False
        self.failure_stage = "event stream"
        self.completed = False
        
        # Task and tools
        self.task = WeaveAgentTask(self, self.name, description)
        self.observation_views = []
        self.tools = {}
        
        # Utilities and context
        self.bm25_index = bm25_index
        self.cache = {}
        self.context = ""

    def run(self):
        """Run the subagent."""
        self.start_time = time.time()
        self.end_time = self.start_time + (self.time_budget * 60)
        while (time.time() < self.end_time) and not self.completed:
            self.tick()
            time.sleep(1)
        return self.completed

    # TODO: Assert that subagent unit test callbacks have names before adding them
    def return_to_caller(self, value: dict):
        """Return thread of execution from subagent to caller. This should be 
        called when the agent's task has been resolved, the task is deemed 
        intractable, or the agent has wandered off so far it can't find 
        its way back to the task."""
        value["name"] = self.name
        value["description"] = self.task.description
        value["children"] = self.children
        
        # Create a schema that includes metadata fields
        result_schema = deepcopy(self.schema)
        result_schema["name"] = {"type": "string"}
        result_schema["description"] = {"type": "string"}
        result_schema["children"] = {"type": "array"}
        result_schema["schema"] = {"type": "object"}
        
        # Add evaluation results
        evaluation_results = self.task.run_evaluations()
        for callback_name, result in evaluation_results.items():
            value[callback_name] = result
            result_schema[callback_name] = {"type": ["boolean", "integer", "float", "string"]}
            
        # Include schema in result
        value["schema"] = result_schema
        
        # Validate against schema
        validate(instance=value, schema=result_schema)
        
        # Setting this interrupts the inference loop and signals an exit
        self.completed = value

    def add_action(self, title: str, callback: Callable) -> None:
        """Add an action callback to the current tick.
        
        Args:
            title: Title of the action
            callback: Function to execute
            
        Raises:
            AssertionError: If title isn't a string or callback isn't a function
        """
        assert isinstance(title, str), "Action title must be a string"
        assert isinstance(callback, types.FunctionType), "Action callback must be a function"
        
        self.current_tick.action = {
            "type": "action",
            "title": title,
            "callback": callback
        }

    def add_observation_view(self, title: str, callback: Callable, tool: Optional[str] = None) -> None:
        """Add an observation callback.
        
        Args:
            title: Title of the observation
            callback: Function to call for observation
            tool: Optional tool name associated with this observation
            
        Raises:
            ValueError: If too many observation views exist
            AssertionError: If callback isn't a function
        """
        # Prevent observation spam
        if len(self.observation_views) > 8:
            raise ValueError(
                "You can't have more than eight observation callbacks at once. "
                "This prevents overwhelming the agent with too much information. "
                "Remove an existing observation first."
            )
            
        assert isinstance(callback, (types.FunctionType, types.MethodType)), \
               "Callback must be a function or method"
               
        view = {
            "type": "observation",
            "title": title,
            "tool": tool,
            "callback": callback
        }
        
        self.observation_views.append(view)

    def remove_observation_view(self, view_title: str) -> None:
        """Remove an observation view by title.
        
        Args:
            view_title: Title of the view to remove
            
        Raises:
            ValueError: If trying to remove a view associated with a tool
        """
        views = [view for view in self.observation_views if view['title'] == view_title]
        
        for view in views:
            # Prevent accidental removal of tool-associated views
            if "tool" in view and view["tool"] in self.tools:
                raise ValueError(
                    f"{view_title} is associated with the {view['tool']} tool. "
                    "Removing it may break tool functionality."
                )
            
            self.observation_views.remove(view)

    # Cache management
    def update_cache(self, key: str, value: Any) -> None:
        """Store a value in the agent's cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        self.cache[key] = value

    def get_cache(self, key: str) -> Any:
        """Retrieve a value from the agent's cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found
        """
        return self.cache.get(key)

    def delete_cache(self, key: str) -> None:
        """Delete a value from the agent's cache.
        
        Args:
            key: Cache key to delete
        """
        if key in self.cache:
            del self.cache[key]

    def add_evaluation(self, title: str, callback: Callable) -> None:
        """Add an evaluation callback to the current tick.
        
        Args:
            title: Title of the evaluation
            callback: Function to execute for evaluation
            
        Raises:
            AssertionError: If title isn't a string or callback isn't a function
        """
        assert isinstance(title, str), "Evaluation title must be a string"
        assert isinstance(callback, types.FunctionType), "Evaluation callback must be a function"
        
        self.current_tick.evaluations.append({
            "type": "evaluation",
            "title": title,
            "callback": callback
        })

    def render_context(self) -> None:
        """Update the agent's context from the tree's context renderer."""
        self.context = self.tree.render_context()

    def generate_block(self, block_type: str, context: str, eval_questions: List[str], 
                      weave_params: Dict, hint: str = "") -> Dict:
        """Generate a block and add it to the event stream.
        
        Args:
            block_type: Type of block to generate
            context: Context for generation
            eval_questions: Evaluation questions for block
            weave_params: Parameters for weave search
            hint: Optional hint to guide generation
            
        Returns:
            The generated block
        """
        return generate_block_inner(self, block_type, context, eval_questions, weave_params, hint)

    def add_block(self, block: Dict) -> None:
        """Add a block to the event stream.
        
        Args:
            block: The block to add
        """
        # Add metadata
        block["subagent"] = self.name
        block["time_remaining"] = self.end_time - time.time()
        
        # Add to tree's event stream
        self.tree.add_block(block)
    
    def add_error_block(self, error_message: str) -> None:
        """Add an error block to the event stream.
        
        Args:
            error_message: The error message to include
        """
        self.debugging = True
        error_block = {
            'type': 'error',
            'message': error_message
        }
        self.add_block(error_block)

    def _do_task_reminder_block(self) -> List[Dict]:
        """Generate task reminder blocks.
        
        These blocks provide a view of the current task and problem state
        to help the agent maintain context.
        
        Returns:
            List of task reminder blocks
        """
        task_reminder_body = ""

        try:
            # Display problem map (tree view of all tasks)
            task_reminder_body += "# Problem Map:\n"
            task_reminder_body += ('"""\n' + self.tree.view_board() + '\n"""')
            
        except Exception:
            # Handle corruption in task data
            tb = traceback.format_exc()
            self.failure_stage = "task reminder"
            error_message = (
                "# TASK REMINDERS OFFLINE DUE TO CORRUPTED DATA. DID YOU DIRECTLY\n"
                "# MODIFY TASK ATTRIBUTES? YOU MUST RESOLVE THIS IMMEDIATELY OR\n"
                "# YOU WILL LOSE TRACK OF WHAT YOU'RE DOING. INVESTIGATE agent.tasks\n"
                "# AND ATTRIBUTES ON TASKS INSIDE.\n"
                f'"""{tb}"""'
            )
            self.add_error_block(error_message)
            
        # Format tasks into blocks
        return [{'type': 'task-reminder', 'body': task_reminder_body}]

    def _do_observation_blocks(self) -> List[Dict]:
        """Generate observation blocks from registered observation views.
        
        This executes all observation callbacks and formats their results
        into observation blocks.
        
        Returns:
            List of observation blocks
        """
        observations = []
        
        # Execute all observation callbacks
        for view in self.observation_views:
            try:
                # Either simulate or execute the observation
                if self.planning:
                    observation_result = simulate_observation(self, view)
                else:
                    observation_result = (view['title'], view['callback'](self))
                    
                observations.append(observation_result)
                
            except Exception:
                # Handle observation errors
                tb = traceback.format_exc()
                error_message = f"# Observation callback '{view['title']}' failed:\n\"\"\"{ tb }\"\"\""
                self.add_error_block(error_message)

        # Format observations into blocks
        observation_blocks = [
            {
                'type': 'observation',
                'title': observation[0],
                'body': observation[1]
            } 
            for observation in observations
        ]
        
        return observation_blocks

    def _do_orientation_block(self) -> Optional[Dict]:
        """Generate an orientation block.
        
        Orientation is where the agent analyzes the situation based on observations
        and decides what to do next. This is the main reasoning/planning phase.
        
        Returns:
            Orientation block if successful, None if generation failed
        """
        # Larger token limits for orientation as it's the main reasoning block
        mcts_params = {
            "weave_n_tokens": 256, 
            "weave_budget": 288,
            "weave_round_budget": 96, 
            "weave_n_expand": 32
        }
        
        # Generate the orientation block
        orientation_block = self._do_tick_block(
            "orientation",
            BLOCK_HINTS["orientation"],
            mcts_params
        )
        
        return orientation_block

    def _do_action_callback_setup(self, attempt_index: int) -> bool:
        """Generate an action block and set up its callback.
        
        This generates the action code block and executes it to set up
        the callback for later execution.
        
        Args:
            attempt_index: Current attempt number (for retry messages)
            
        Returns:
            True if successful, False if failed
        """
        # Generate action block
        action_block = self._do_tick_block("action", BLOCK_HINTS["action"], {})
        
        # Check if block generation succeeded
        if not action_block:
            # Failed to generate block
            attempts_remaining = 3 - (attempt_index + 1)
            self.add_error_block(
                f"# Failed to generate action block. {attempts_remaining} attempts remaining."
            )
            return False

        # Store action block in current tick
        self.current_tick.action_setup = action_block

        # Set up action callback
        try:
            if self.planning:
                # In planning mode, just analyze for valid syntax
                setup_placeholder_callbacks(self, action_block['body'])
            else:
                # In execution mode, actually execute the setup code
                exec(action_block['body'])
                
            return True
            
        except Exception:
            # Handle callback setup failure
            tb = traceback.format_exc()
            error_message = f"# Action setup failed:\n\"\"\"{tb}\"\"\""
            self.add_error_block(error_message)
            self.failure_stage = "action"
            
            # Generate debug block to help diagnose the issue
            try:
                self._do_tick_block("debug", BLOCK_HINTS["debug"], {})
            except Exception:
                pass
                
            return False

    def _do_action_callback(self, attempt_index: int) -> Tuple[bool, Any]:
        """Execute the action callback.
        
        This runs the callback that was set up in _do_action_callback_setup.
        
        Args:
            attempt_index: Current attempt number (for retry messages)
            
        Returns:
            Tuple of (success boolean, action result)
        """
        # Run action callback
        try:
            if self.planning:
                # In planning mode, simulate the result
                action_result = None
                simulated_error = roll_for_error_block(self, "# Action execution failed:\n")
                if simulated_error:
                    raise Exception(simulated_error)
            else:
                # In execution mode, actually run the callback
                action_result = self.current_tick.action["callback"](self)
                
            return True, action_result
            
        except Exception:
            # Handle callback execution failure
            if self.planning:
                # In planning mode, use the simulated error
                self.add_error_block(simulated_error)
            else:
                # In execution mode, capture the actual error
                tb = traceback.format_exc()
                error_message = f"# Action execution failed:\n\"\"\"{tb}\"\"\""
                self.add_error_block(error_message)
                
            self.failure_stage = "action"
            action_result = "ERROR"
            
            # Generate debug block to help diagnose the issue
            try:
                self._do_tick_block("debug", BLOCK_HINTS["debug"], {})
            except Exception:
                pass
                
            return False, action_result

    def _do_expectation_block(self) -> Optional[Dict]:
        """Generate an expectation block.
        
        This is where the agent predicts what success and failure would look like,
        to be checked in the evaluation phase.
        
        Returns:
            Expectation block if successful, None otherwise
        """
        expectation_block = self._do_tick_block(
            "expectation",
            BLOCK_HINTS["expectation"],
            {}
        )
        
        return expectation_block

    def _do_observation_inference_block(self) -> Optional[Dict]:
        """Generate an observation inference block.
        
        This block contains code to manage the observation callbacks,
        adding or removing them as needed based on expectations.
        
        Returns:
            Observation inference block if successful, None otherwise
        """
        observation_inference_block = self._do_tick_block(
            "observation-inference",
            BLOCK_HINTS["observation-inference"],
            {}
        )
        
        return observation_inference_block

    def _do_observation_updates(self) -> bool:
        """Execute the observation inference block to update observation callbacks.
        
        This executes the code in the observation inference block to add or
        remove observation callbacks.
        
        Returns:
            True if successful, False if failed
        """
        try:
            if self.planning:
                # In planning mode, just check syntax
                setup_placeholder_callbacks(self, self.current_tick.observation_inference['body'])
            else:
                # In execution mode, actually update observations
                exec(self.current_tick.observation_inference['body'])
                
            return True
            
        except Exception:
            # Handle execution failure
            tb = traceback.format_exc()
            error_message = f"# Observation inference execution failed:\n\"\"\"{tb}\"\"\""
            self.add_error_block(error_message)
            self.failure_stage = "observation-inference"
            
            return False

    def _do_evaluation_block(self, attempt_index: int) -> Optional[Dict]:
        """Generate an evaluation block.
        
        This block contains callbacks to check if the action succeeded
        according to the expectations.
        
        Args:
            attempt_index: Current attempt number (for retry messages)
            
        Returns:
            Evaluation block if successful, None if failed
        """
        eval_block = self._do_tick_block(
            "evaluation",
            BLOCK_HINTS["evaluation"],
            {}
        )
        
        if eval_block:
            return eval_block
        else:
            # Failed to generate block
            attempts_remaining = 3 - (attempt_index + 1)
            
            # Try to generate a debug block to diagnose the issue
            try:
                self._do_tick_block("debug", BLOCK_HINTS["debug"], {})
            except Exception:
                pass
                
            self.add_error_block(
                f"# Failed to generate evaluation block. {attempts_remaining} attempts remaining."
            )
            
            return None

    def _do_evaluation_callback_setup(self, attempt_index: int, eval_block: Dict) -> bool:
        """Set up evaluation callbacks from an evaluation block.
        
        Args:
            attempt_index: Current attempt number (for retry messages)
            eval_block: The evaluation block to process
            
        Returns:
            True if successful, False if failed
        """
        try:
            if self.planning:
                # In planning mode, just check syntax
                setup_placeholder_callbacks(self, eval_block['body'])
            else:
                # In execution mode, actually set up callbacks
                exec(eval_block['body'])
                
            return True
            
        except Exception:
            # Handle setup failure
            tb = traceback.format_exc()
            error_message = f"# Evaluation setup failed:\n\"\"\"{tb}\"\"\""
            self.add_error_block(error_message)
            self.failure_stage = "evaluation"
            
            # Try to generate a debug block
            try:
                self._do_tick_block("debug", BLOCK_HINTS["debug"], {})
            except Exception:
                pass
                
            return False

    def _do_evaluation_callbacks(self) -> List[List]:
        """Execute all evaluation callbacks.
        
        This runs all the evaluation callbacks that were set up in the
        evaluation block and collects their results.
        
        Returns:
            List of [evaluation_title, result] pairs
        """
        action_evaluation_results = []
        
        for evaluation in self.current_tick.evaluations:
            try:
                if self.planning:
                    # In planning mode, simulate result
                    result = None
                    simulated_error = roll_for_error_block(self, "# Evaluation failed: \n")
                    if simulated_error:
                        raise Exception(simulated_error)
                else:
                    # In execution mode, actually run callback
                    result = evaluation["callback"](self)
                    
                # Stringify complex result types for JSON serialization
                if type(result) not in [str, int, bool, float, type(None)]:
                    result = repr(result)
                    
                action_evaluation_results.append([evaluation['title'], result])
                
            except Exception:
                # Handle evaluation failure
                if self.planning:
                    self.add_error_block(simulated_error)
                else:
                    tb = traceback.format_exc()
                    error_message = f"# Evaluation failed:\n\"\"\"{tb}\"\"\""
                    self.add_error_block(error_message)
                    
                action_evaluation_results.append([evaluation['title'], "ERROR"])
                
        return action_evaluation_results
        
    def _do_tick_block(self, block_type: str, hint: str, wp_update: Dict) -> Optional[Dict]:
        """Generate a block of a specific type.
        
        This is a common utility method used by all the _do_*_block methods
        to generate blocks using the appropriate parameters and error handling.
        
        Args:
            block_type: Type of block to generate
            hint: Hint string to guide generation
            wp_update: Updates to weave parameters
            
        Returns:
            Generated block if successful, None if failed
        """
        # Base weave search parameters
        weave_params = {
            "weave_n_tokens": 256, 
            "weave_budget": 72,
            "weave_round_budget": 24, 
            "weave_n_expand": 16,
            "weave_beam_width": 1, 
            "weave_max_lookahead": 3,
            "weave_temperature": 0.2
        }
        
        # Apply custom parameters
        weave_params.update(wp_update)
        
        # Load evaluation questions for this block type
        try:
            eval_file_path = f"/app/eval_rubrics/{block_type}.txt"
            with open(eval_file_path) as infile:
                inference_questions = infile.read().strip().splitlines()
        except FileNotFoundError:
            inference_questions = []
            self.add_error_block(f"# Missing evaluation questions file for {block_type}")
        
        # Log block generation
        rprint(f"Writing block #[cyan]{self.tree.current_block_index()}[/cyan] of type [cyan]{block_type}[/cyan]")
        
        # Generate the block
        try:
            block = self.generate_block(
                block_type,
                self.context,
                inference_questions,
                weave_params,
                hint=hint
            )
            
            # Update context after generation
            self.render_context()
            
            return block
            
        except ValueError:
            # Handle block generation failure
            tb = traceback.format_exc()
            
            error_message = (
                "# Block generation failed. Common issues include:\n"
                "- Incorrect callback structure (should be 'def name(agent): ...')\n"
                "- Undefined variables or imports\n"
                "- Syntax errors\n"
                f"\n\"\"\"{tb}\"\"\""
            )
            
            self.add_error_block(error_message)
            self.failure_stage = block_type
            
            return None
    
    def tick(self) -> None:
        """Execute one complete cognitive cycle (tick).
        
        A tick consists of the following phases:
        1. Task reminders and observations
        2. Orientation reasoning
        3. Action generation and execution
        4. Expectation formation
        5. Observation inference and updates
        6. Evaluation setup and execution
        7. Outcome recording
        
        The tick will exit early if any phase fails.
        """
        # Check if previous tick had errors
        try:
            if hasattr(self.current_tick, 'outcome') and "ERROR" in [
                outcome[1] for outcome in self.current_tick.outcome["table"]
            ]:
                self.debugging = True
        except (AttributeError, KeyError):
            self.debugging = True
            
        # Create new tick
        self.current_tick = Tick(self, len(self.ticks))

        # Phase 1: Generate task reminders and observations
        task_blocks = self._do_task_reminder_block()
        observation_blocks = self._do_observation_blocks()

        # Add blocks to event stream
        for new_block in (task_blocks + observation_blocks):
            self.add_block(new_block)
            
        # Update context with new blocks
        self.render_context()

        # Save state to disk (for reproducibility and debugging)
        self.tree.dump_event_stream()
            
        # Phase 2: Generate orientation block
        orientation_block = self._do_orientation_block()
        if not orientation_block:
            return  # Exit if orientation failed
            
        self.current_tick.orientation = orientation_block

        # Phase 3: Action generation and execution
        action_failed = True
        for attempt in range(3):  # Allow up to 3 attempts
            # Try to set up action callback
            is_action_setup = self._do_action_callback_setup(attempt)
            if not is_action_setup:
                continue
                
            # Try to execute action callback
            is_action_executed, action_result = self._do_action_callback(attempt)
            if is_action_executed:
                action_failed = False
                break
                
        # Exit if action setup or execution failed after all attempts
        if not hasattr(self.current_tick, "action_setup") or action_failed:
            return
        
        # Phase 4: Generate expectation block
        expectation_block = self._do_expectation_block()
        if not expectation_block:
            return  # Exit if expectation failed
            
        self.current_tick.expectation = expectation_block
            
        # Phase 5: Generate observation inference block
        observation_inference_block = self._do_observation_inference_block()
        if not observation_inference_block:
            return  # Exit if observation inference failed
            
        self.current_tick.observation_inference = observation_inference_block

        # Execute observation updates
        are_observations_updated = self._do_observation_updates()
        if not are_observations_updated:
            return  # Exit if observation updates failed
        
        # Phase 6: Evaluation generation and execution
        evaluation_blocks = []
        eval_failed = True
        
        for attempt in range(3):  # Allow up to 3 attempts
            # Try to generate evaluation block
            eval_block = self._do_evaluation_block(attempt)
            if not eval_block:
                continue
                
            # Try to set up evaluation callbacks
            is_evaluation_setup = self._do_evaluation_callback_setup(attempt, eval_block)
            if not is_evaluation_setup:
                continue
                
            evaluation_blocks.append(eval_block)
            eval_failed = False
            break
            
        # Exit if evaluation failed after all attempts
        if eval_failed:
            return
            
        self.current_tick.evaluation_setup = evaluation_blocks

        # Phase 7: Execute evaluations and record outcomes
        
        # Run task evaluation callbacks
        task_evaluation_results = []
        for evaluation in self.task.evaluations:
            try:
                if self.planning:
                    result = None
                else:
                    result = evaluation["callback"](self)
                task_evaluation_results.append([evaluation['title'], result])
            except Exception:
                tb = traceback.format_exc()
                task_evaluation_results.append([evaluation['title'], "ERROR"])

        # Run action evaluation callbacks
        action_evaluation_results = self._do_evaluation_callbacks()

        # Combine all results
        outcomes = []
        
        # Add action result
        try:
            if self.planning:
                outcomes.append([self.current_tick.action["title"], None])
            else:
                outcomes.append([self.current_tick.action["title"], action_result])
        except AttributeError:
            outcomes.append(("[No action specified with agent.add_action()]", "ERROR"))
            
        # Add all evaluation results
        outcomes.extend(task_evaluation_results)
        outcomes.extend(action_evaluation_results)
        
        # Create and add outcome block
        outcome_block = {
            'type': 'outcome',
            "subagent": self.name,
            "index": self.tree.current_block_index() + 1,
            "timestamp": time.time(),
            "time_remaining": self.end_time - time.time(),
            'table': outcomes
        }
        
        # Simulate outcomes if in planning mode
        if self.planning:
            outcome_block = simulate_outcomes(self.model_name, outcome_block)
            
        # Add to event stream
        self.add_block(outcome_block)
        self.current_tick.outcome = outcome_block
        
        # Validate tick (in execution mode only)
        try:
            if not self.planning:
                self.current_tick.validate()
        except Exception:
            tb = traceback.format_exc()
            error_message = f"# Tick validation failed:\n\"\"\"{tb}\"\"\""
            self.add_error_block(error_message)
            self.current_tick.is_valid = False
            
        # Store completed tick and reset state
        self.ticks.append(self.current_tick)
        self.debugging = False
        self.failure_stage = "event stream"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", help="The model to use.")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer to use (if different from model_name)")
    parser.add_argument("--port", default=5000, help="The port to use for VLLM.")
    parser.add_argument("--bootstrap",
                        default="bootstrap.py",
                        help="The filepath to run as bootstrap.")
    parser.add_argument("--budget", type=int, default=360,
                        help="Time budget for the run in minutes.")
    args = parser.parse_args()
        
    def simple_evaluate_outputs(score_prompt_fns, texts):
        if type(texts) == str:
            texts = [texts,]
        if type(score_prompt_fns) in [types.FunctionType, functools.partial]:
            score_prompt_fns = [score_prompt_fns,]
        scores = asyncio.run(evaluate_outputs_vllm(args.model_name,
                                                   score_prompt_fns,
                                                   texts,
                                                   port=args.port))
        return torch.sigmoid(scores)

    def simple_bayes_evaluate_outputs(parent_q, questions, texts):
        if type(texts) == str:
            texts = [texts,]
        score_prompt_fns = [make_simple_bayes_score_prompt(question)
                            for question in questions]
        scores = asyncio.run(bayesian_evaluate_outputs_vllm(args.model_name,
                                                            parent_q,
                                                            score_prompt_fns,
                                                            texts,
                                                            port=args.port))
        return scores

    
    agent = WeaveAgentTree(args.model_name, args.budget)

    if not args.tokenizer:
        args.tokenizer = args.model_name

    with open("hf_token.txt") as infile:
        os.environ["HF_TOKEN"] = infile.read().strip()
    # Delete token so it doesn't leak into traces
    os.remove("hf_token.txt")
    agent.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    schema_builder = SchemaBuilder()
    schema_builder.add_text_field("id", stored=True, tokenizer_name='raw')
    schema_builder.add_text_field("type", stored=True)
    schema_builder.add_text_field("render", stored=True)
    schema_builder.add_text_field("q", stored=True)
    schema_builder.add_float_field("score", stored=True)
    schema_builder.add_integer_field("index", stored=True)
    schema_builder.add_float_field("timestamp", stored=True)
    schema_builder.add_text_field("description", stored=True)

    bm25_schema = schema_builder.build()

    if not os.path.exists("memories"):
        os.mkdir("memories")
    if not os.path.exists("memories/bm25"):
        os.mkdir("memories/bm25")
    bm25_index = Index(bm25_schema, path="./memories/bm25")
    
    # Mock bootstrap agent so we can run the callbacks in bootstrap file
    self = agent.subagent(
        "bootstrap",
        None,
        "Bootstrap the weave-agent",
        {},
        args.budget,
                          
    )
    with open("weave_agent.py") as infile:
        # Genesis block
        genesis_block = {
            'type': 'genesis',
            'body': infile.read()
        }
        self.add_block(genesis_block)

    with open(args.bootstrap) as infile:
        # Bootstrap block
        bootstrap_block = {
            'type': 'bootstrap',
            'body': infile.read()
        }
        self.add_block(bootstrap_block)
        exec(bootstrap_block["body"])

    def run_bootstrap_callbacks(subagent):
        """Run bootstrap callbacks in function to avoid contaminating global scope."""
        # Run action callback
        action_result = subagent.current_tick.action["callback"](subagent)

        # Run evaluation callbacks
        evaluation_results = []
        for evaluation in subagent.current_tick.evaluations:
            result = evaluation["callback"](subagent)
            evaluation_results.append((evaluation['title'], result))

        outcomes =  []
        outcomes += [(subagent.current_tick.action["title"],action_result),]
        outcomes += evaluation_results

        # Add outcome block
        outcome_block = {
            'type': 'outcome',
            'table': outcomes
        }
        subagent.add_block(outcome_block)
        subagent.current_tick.outcome = outcome_block

    run_bootstrap_callbacks(self)
    # Clean up mock bootstrap agent
    del(self)

    if not os.path.exists("/app/weave-agent-logs"):
        os.mkdir("/app/weave-agent-logs")
        
    result, event_stream = agent.run("main")
    
    with open(f"/app/weave-agent-logs/{round(time.time())}/log.json", "w") as outfile:
        out = {"model_name":args.model_name,
               "event_stream":event_stream,
               "result":result,}
        json.dump(out, outfile)
        outfile.flush()
