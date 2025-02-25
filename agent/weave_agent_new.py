# hint Listen Carefully: This program is self modifying code.
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

# SUBAGENT HIERARCHY AND TOOL INTEGRATION:
# The Weave Agent framework uses a hierarchical structure of subagents to solve complex problems:
#
# 1. Subagent Structure:
#    - Root agent ("main") receives the initial task
#    - Root agent can create child subagents for subtasks
#    - Each subagent has its own time budget and return schema
#    - Subagents can further delegate to their own children
#    - Results flow back up the tree with schema validation
#
# 2. Tool Integration:
#    - External tools (like editor, discord, nethack) plug into the agent
#    - Tools are registered in the agent.tools dictionary
#    - Tools provide observation views showing their current state
#    - Tools offer methods the agent can call during action phases
#    - This pattern allows integration with virtually any API or service

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


class WeaveAgentTask:
    def __init__(self, subagent, title: str, description: str = ""):
        self.subagent = subagent
        self.title = str(title)
        self.description = description
        self.evaluations = []

    def add_evaluation(self, title, callback):
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.evaluations.append(
            {"type": "evaluation", "title": title, "callback": callback}
        )

    def run_evaluations(self):
        results = {}
        for evaluation in self.evaluations:
            try:
                result = evaluation["callback"](self.subagent)
            except Exception:
                result = traceback.format_exc()
            results[evaluation["callback"].__name__] = result
        return results


class BlockType(Enum):
    OBSERVATION = auto()
    TASK_REMINDER = auto()
    ORIENTATION = auto()
    ACTION = auto()
    ERROR = auto()
    DEBUG = auto()
    EXPECTATION = auto()
    OBSERVATION_INFERENCE = auto()
    EVALUATION = auto()
    OUTCOME = auto()


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
    def __init__(self, model_name: str, time_budget: int):
        """
        Initialize the root of the agent tree structure.

        Args:
            model_name: Name of the LLM to use for generating blocks
            time_budget: Total time in minutes allocated to the agent
        """
        self.model_name = model_name
        self.__agents = {}
        self.__time_budget = time_budget
        # Pin genesis and bootstrap so agent knows how to use framework
        self.__pinned_events = [0, 1]
        self.__current_block_index = 0
        self.__event_stream = []
        self.transitions = {
            BlockType.OBSERVATION: [BlockType.OBSERVATION, BlockType.ORIENTATION],
            BlockType.TASK_REMINDER: [BlockType.OBSERVATION, BlockType.ORIENTATION],
            BlockType.ORIENTATION: [BlockType.ACTION, BlockType.ERROR],
            BlockType.ACTION: [BlockType.EXPECTATION, BlockType.ERROR],
            BlockType.ERROR: [
                BlockType.DEBUG,
                BlockType.ACTION,
                BlockType.EVALUATION,
                BlockType.OUTCOME,
                BlockType.TASK_REMINDER,
                BlockType.ERROR,
            ],
            BlockType.DEBUG: [
                BlockType.ACTION,
                BlockType.EVALUATION,
                BlockType.TASK_REMINDER,
                BlockType.ERROR,
            ],
            BlockType.EXPECTATION: [
                BlockType.OBSERVATION_INFERENCE,
                BlockType.TASK_REMINDER,
                BlockType.ERROR,
            ],
            BlockType.OBSERVATION_INFERENCE: [
                BlockType.EVALUATION,
                BlockType.ERROR,
                BlockType.TASK_REMINDER,
            ],
            BlockType.EVALUATION: [BlockType.OUTCOME, BlockType.ERROR],
            BlockType.OUTCOME: [BlockType.TASK_REMINDER, BlockType.ERROR],
        }

    def run(self, name):
        """
        Execute the named subagent until completion or timeout.

        This is the main entry point for running a subagent task.

        Args:
            name: Name of the subagent to run

        Returns:
            The result returned by the subagent
        """
        import time

        start_time = time.time()
        deadline = float(self.__agents[name].end_time)
        return_schema = deepcopy(self.__agents[name].schema)
        result = self.__agents[name].run()
        validate(instance=result, schema=return_schema)
        end_time = time.time()
        if end_time > deadline + 300:
            # TODO: More nuanced way to handle this
            raise ValueError("Time exceeded!")
        else:
            return result

    def subagent(self, name, parent, description, schema, time_budget):
        """
        Create a new subagent in the agent tree.

        This is the main method for delegating tasks to subtask-specific agents.
        Each subagent has its own tick cycle and tools, but shares the overall context.

        Args:
            name: Unique identifier for the subagent
            parent: Parent agent name (None for root)
            description: Human-readable task description
            schema: JSON schema defining the expected return structure
            time_budget: Time in minutes allocated to this task

        Returns:
            The newly created WeaveAgentNode
        """
        if name in self.__agents:
            raise ValueError
        reserved_words = {"name", "description", "children", "schema"}
        assert not set(schema).intersection(reserved_words)
        if parent:
            self.__agents[parent].children.append(name)
        try:
            subagent = WeaveAgentNode(
                self, parent, name, description, schema, time_budget
            )
        except Exception as e:
            self.__agents[parent].children.remove(name)
            raise e
        self.__agents[name] = subagent
        return subagent

    def is_valid_transition(self, next_block_type):
        """
        Check if a transition to the given block type is valid.

        The agent follows a state machine that enforces the correct sequence
        of cognitive operations. This validates that the agent doesn't skip
        important steps or perform them in the wrong order.

        Args:
            next_block_type: The type of block to transition to

        Returns:
            True if transition is valid

        Raises:
            ValueError if transition is invalid
        """
        if type(next_block_type) == str:
            try:
                next_block_type = getattr(
                    BlockType, next_block_type.upper().replace("-", "_")
                )
            except AttributeError:
                raise ValueError(f"Unknown block type: {next_block_type}")
        if self.__event_stream[-1]["type"] in {"genesis", "bootstrap"}:
            return True
        else:
            current_state = getattr(
                BlockType, self.__event_stream[-1]["type"].upper().replace("-", "_")
            )
        if next_block_type in self.transitions.get(current_state, []):
            return True
        else:
            raise ValueError(
                f"Invalid transition from {current_state} to {next_block_type}"
            )

    def add_block(self, block):
        """
        Add a new block to the agent's event stream and memory index.

        This method:
        1. Validates that the block follows the correct state transition
        2. Assigns metadata (timestamp, index, etc.)
        3. Generates a description if not provided
        4. Adds the block to both the event stream and BM25 index

        The BM25 index serves as searchable memory, allowing the agent to
        find relevant past blocks based on semantic similarity.

        Args:
            block: Dictionary containing block data
        """
        if block["type"] not in {"genesis", "bootstrap"}:
            self.is_valid_transition(block["type"])
        block["index"] = self.__current_block_index
        block["timestamp"] = time.time()
        if block["type"] == "orientation":
            block["metadata"] = {
                "block_index": self.__current_block_index,
                "working_directory": os.getcwd(),
            }
        if "q" not in block:
            block["q"] = ""
        if "score" not in block:
            # TODO: Make actual score function for observations, task reminders etc
            block["score"] = 2
        # TODO: Make these parallel requests
        # TODO: Add view to tuner for training the descriptions
        render = render_block(block)
        if "description" not in block:
            with open("/app/templates/describe1.txt") as infile:
                template = infile.read()
                prompt = template.format(rendered_block=render)
                object_description = generate_outputs_vllm(
                    self.model_name,
                    prompt,
                    512,
                    port=args.port,
                    n=1,
                    stop=[
                        "</summary>",
                    ],
                )[0]
            with open("/app/templates/describe2.txt") as infile:
                template = infile.read()
                context = self.render_context()
                prompt = template.format(
                    rendered_block=render,
                    object_description=object_description,
                    rendered_context=context,
                )
                context_description = generate_outputs_vllm(
                    self.model_name,
                    prompt,
                    512,
                    port=args.port,
                    n=1,
                    stop=[
                        "</summary>",
                    ],
                )[0]
            # TODO: Make actual tagging function
            block["description"] = object_description + "\n\n" + context_description
        self.__event_stream.append(block)

        if block["type"] not in {"genesis", "bootstrap"}:
            block_render = render_block(block)
            sha256_hash = hashlib.sha256()
            sha256_hash.update(block_render.encode("utf-8"))
            hash_hex = sha256_hash.hexdigest()
            writer = bm25_index.writer()
            writer.add_document(
                tantivy.Document(
                    id=hash_hex,
                    type=block["type"],
                    render=block_render,
                    q=block["q"],
                    score=block["score"],
                    index=block["index"],
                    timestamp=block["timestamp"],
                    description=block["description"],
                )
            )
            writer.commit()

        self.__current_block_index += 1

    def current_block_index(self):
        return self.__current_block_index

    def find_last_block_of_type(self, _type):
        """Get the last block of a particular type, if none in trace return none."""
        for block in reversed(self.__event_stream):
            if block["type"] == _type:
                return block
        return None

    def render_context(self):
        context = ""
        context_blocks = []
        history_len = 60
        for index in self.__pinned_events:
            if (len(self.__event_stream) - index) > history_len:
                context_blocks.append(self.__event_stream[index])
        context_blocks += self.__event_stream[-history_len:]
        for event_block in context_blocks:
            context += render_block(event_block)
        return context

    def view_board(self, root="main") -> str:
        """
        Generate a visualization of the current agent hierarchy.

        Creates a nested dictionary showing the agent tree structure with
        details about each agent's state, evaluations, and children.

        Args:
            root: Name of the root agent to start visualization from

        Returns:
            String representation of the agent tree
        """
        problem_map = {}
        substack = [
            root,
        ]
        while substack:
            subagent = self.__agents[substack.pop()]
            parent = subagent.name
            path = []
            while parent:
                path.append(parent)
                # Convert to object so we can get grandparent
                parent = self.__agents[parent]
                parent = parent.parent
            path.reverse()
            current_level = problem_map
            for key in path:
                if key not in current_level:
                    current_level[key] = {}
                current_level = current_level[key]
            current_level["name"] = subagent.name
            current_level["description"] = subagent.task.description
            current_level["evaluations"] = subagent.task.run_evaluations()
            current_level["time_remaining"] = subagent.end_time - time.time()
            current_level["completed"] = subagent.completed
            current_level["schema"] = subagent.schema
            substack.extend(subagent.children)
        return pformat(problem_map)

    def dump_event_stream(self):
        with open(
            f"/app/weave-agent-logs/event_trace_{round(time.time())}.json", "w"
        ) as outfile:
            json.dump(self.__event_stream, outfile)
        with open(
            f"/app/weave-agent-logs/rendered_trace_{round(time.time())}.py", "w"
        ) as outfile:
            for event_block in self.__event_stream:
                outfile.write(render_block(event_block))
            outfile.flush()


class Tick:
    """
    Represents a complete execution cycle of the agent.

    A tick contains all blocks that make up one cognitive cycle:
    - Orientation (thinking about observations)
    - Action (executing code)
    - Expectation (predicting outcomes)
    - Evaluations (checking results)
    - Outcome (summarizing what happened)

    Ticks are the fundamental unit of execution in the Weave Agent.
    """

    def __init__(self, agent, index):
        self._agent = agent
        self.tick_id = index
        self.evaluations = []

    def validate(self):
        """
        Ensure this tick has all required components.

        A valid tick must have orientation, action, expectation,
        at least one evaluation, and an outcome.

        Raises:
            ValueError if any required component is missing
        """
        if not hasattr(self, "orientation"):
            raise ValueError("No orientation on tick.")
        elif not hasattr(self, "action"):
            raise ValueError("No action on tick.")
        elif "body" not in self.action_setup:
            raise TypeError("Tick action has no program.")
        elif not hasattr(self, "expectation"):
            raise ValueError("No expectation on tick.")
        elif not self.evaluations:
            raise ValueError("No evaluations on tick.")
        elif not hasattr(self, "outcome"):
            raise ValueError("No outcome on tick.")

    def to_json(self):
        """Convert tick data to JSON-serializable format."""
        return {
            "tick_id": self.tick_id,
            "orientation": self.orientation,
            "action": repr(self.action),
            "expectation": self.expectation,
            "evaluations": repr(self.evaluations),
            "outcome": repr(self.outcome),
        }


# The intended problem solving strategy for subagents is to delegate until you
# reach a base case that can be solved in a short number of actions and then
# resolve it. The root task is allocated a certain amount of time which it can
# then delegate to subagent calls. Remember not to allocate all of the available
# time to a call tree unless you're very rushed, you should assume there will be
# failures and budget tasks the time that they need rather than just splitting
# up the available time between them.
class WeaveAgentNode:
    def __init__(self, tree, parent, subagent_name, description, schema, time_budget):
        """
        Initialize a Weave Agent node that can execute a task.

        Args:
            tree: The parent WeaveAgentTree that manages all nodes
            parent: The name of the parent agent (None for root)
            subagent_name: Unique identifier for this agent node
            description: Human-readable description of this agent's task
            schema: JSON schema defining the expected return value format
            time_budget: Time in minutes allocated to this task
        """
        self.tree = tree
        self.parent = parent
        self.children = []
        self.model_name = self.tree.model_name
        self.name = subagent_name
        self.schema = schema
        self.creation_time = time.time()
        self.time_budget = time_budget
        self.end_time = self.creation_time + (time_budget * 60)
        self.current_tick = Tick(self, 0)
        self.ticks = []
        # TODO: Remove this once done testing
        self.planning = True
        self.debugging = False
        self.failure_stage = "event stream"
        self.task = WeaveAgentTask(self, self.name, description)
        self.observation_views = []
        # TODO: Do I really need to have this pointer?
        self.bm25_index = bm25_index
        self.tools = {}
        self.cache = {}
        self.context = ""
        self.completed = False

        # Centralized hint text - restored to original verbose version
        self.hints = {
            "orientation": (
                "#hint The orientation block is my opportunity to\n"
                + "# reflect on the situation, do chain of thought,\n"
                + "# summarize what has happened and what needs to\n"
                + "# be done in response, etc. It is only technically\n"
                + "# python code and does not get executed by the\n"
                + "# framework. I suggest putting my internal\n"
                + "# monologue in a triple quote block at this step.\n"
                + "# Orientation uses the MiniHF Morpheus format. Morpheus\n"
                + "# solves problems through discussion between personas\n"
                + "# or postures representing different aspects of weave-agent\n"
                + "# such as mental motions, perspectives on the problem, etc.\n"
                + "# The first posture is always expectation because at the\n"
                + "# start of a tick we evaluate whether the expectation we\n"
                + "# formed about the action taken in the last tick was\n"
                + "# violated or not. The different personas then discuss\n"
                + "# what to do in light of this. Some postures weave-agent\n"
                + "# has include:\n"
                + "#\n"
                + "# WEAVER [P: EXPECTATION], I analyze whether the expectation\n"
                + "# was met or not by the observable results of the previous\n"
                + "# action.\n"
                + "#\n"
                + "# WEAVER [P: HYPOTHESIS], I enumerate different hypothesis\n"
                + "# and point out ways we could gain more information about\n"
                + "# which of them is true.\n"
                + "#\n"
                + "# WEAVER [P: RATIONAL], I focus on inferences we can make\n"
                + "# by employing first principles reasoning or logical\n"
                + "# extrapolation from well known mental models and premises.\n"
                + "#\n"
                + "# WEAVER [P: EMPIRICISM], I focus on inferences we can make\n"
                + "# by paying attention to sensory observations and concrete\n"
                + "# examples. I have a habit of pointing out when an extrapolation\n"
                + "# from RATIONAL is contradicted by an observable phenomenon\n"
                + "# or piece of evidence from the world. We then reconcile\n"
                + "# the contradiction together.\n"
                + "#\n"
                + "# WEAVER [P: RATIONAL], We do actually discuss things by the\n"
                + "# way.\n"
                + "#\n"
                + "# WEAVER [P: EMPIRICISM], As you could have inferred from the\n"
                + "# description of the Morpheus format above this conversation,\n"
                + "# yes. Let's continue.\n"
                + "#\n"
                + "# WEAVER [P: ARBITER], I coordinate the discussion and help\n"
                + "# resolve disputes that arise between weave-agent's personas.\n"
                + "# I'm especially likely to appear if things are starting to\n"
                + "# get overly rude or derail.\n"
                + "#\n"
                + "# WEAVER [P: ARBITER], By the way a posture can talk twice in\n"
                + "# a row if it has meaningfully separate thoughts about\n"
                + "# something and it would make the most ergonomic sense to\n"
                + "# separate them.\n"
                + "#\n"
                + "# WEAVER [P: RATIONAL-2], Postures can also talk to themselves\n"
                + "# if their thought comes from the same emotional-cognitive place.\n"
                + "#\n"
                + "# WEAVER [P: RATIONAL-1], Yeah but I don't have anything to say\n"
                + "# to myself right now so introduce the next guy.\n"
                + "#\n"
                + "# WEAVER [P: CONCLUSION], I appear at the end of the discussion\n"
                + "# to write the concluding block outlining our next steps as a\n"
                + "# bullet point list. Speaking of which, it's time to get started!\n"
            ),
            "action": (
                "#hint Action blocks are where I write code to take actions.\n"
                + "# If the task makes sense to break into parts, define subagents\n"
                + "# to delegate to using agent.subagent(). Make sure to define a\n"
                + "# schema and task evaluations for each subagent. If it won't fit\n"
                + "# into one action block keep in mind you can define subagents \n"
                + "# across multiple blocks and then do agent.run() to execute them.\n"
                + "# If it seems possible to resolve the current task as a base case\n"
                + "# in a handful of actions then write a callback to further my goal(s)\n"
                + "# based on the orientation block and set up the callback to be\n"
                + "# executed with the self.add_action() method. I must write a \n"
                + "# callback and then set it up to be executed\n"
                + "# later with self.add_action() or the tick will not be accepted.\n"
                + "# It's important to remember that my callback can do anything\n"
                + "# a python program can do through side effects in the external\n"
                + "# computable environment. If I need to import a new module make sure\n"
                + "# to do it inside the callback because the tick gets executed in a\n"
                + "# local context."
            ),
            "expectation": (
                "#hint Expectation blocks are where I think about what it would\n"
                + "# look like for my action to succeed, what it would look like\n"
                + "# for it to fail. I am enumerating the expected sensory evidence\n"
                + "# that would tell me one way or another whether my action is\n"
                + "# working or not. Like the orientation this should go in triple\n"
                + "# quotes."
            ),
            "observation_inference": (
                "# In the observation inference stage I manage the observation\n"
                + "# callbacks that fetch information on each tick. Since I just\n"
                + "# formulated my expectations now is my opportunity to review\n"
                + "# and change the observation blocks that will be presented on the\n"
                + "# next tick. I should avoid redundant observation callbacks. I\n"
                + "# can remove ones that are no longer necessary or mostly distracting\n"
                + "# with remove_observation_view(view_title). If new callbacks seem useful\n"
                + "# to help me orient and judge whether the action had the intended\n"
                + "# side effects on the computable environment I can add them\n"
                + "# with add_observation_view(title, callback)"
            ),
            "evaluation": (
                "#hint Evaluation blocks are where I write callbacks to check if\n"
                + "# my action succeeded or not based on the expectation. There are\n"
                + "# unit tests and logit evaluators. Use unit test callbacks\n"
                + "# (i.e. normal python) for symbolic manipulation tasks like\n"
                + "# checking arithmetic, the existence of a particular file, etc.\n"
                + "# Use logit evaluators for vibe-y tasks like whether a piece of\n"
                + "# writing flows well or if a source seems trustworthy. Like\n"
                + "# reminders both unit test callbacks and logit evaluators return\n"
                + "# a value between 0 and 1. I should be sure to add my callback to\n"
                + "# the queue with agent.add_evaluation(title, callback)."
            ),
            "debug": (
                "#hint Debug blocks are my opportunity to reason about the failure\n"
                "# I just experienced. Because I get multiple opportunities to\n"
                "# take an action before I'm booted to the next orientation stage\n"
                "# I can formulate hypothesis and use the next action blocks to test them.\n"
                "# I want to narrow in on the cause of failure and take steps to resolve\n"
                "# the issue."
            ),
        }

    def run(self):
        """Run the subagent."""
        self.start_time = time.time()
        self.end_time = self.start_time + (self.time_budget * 60)
        while (time.time() < self.end_time) and not self.completed:
            self.tick()
            time.sleep(1)
        return self.completed

    def return_to_caller(self, value: dict):
        """
        Return execution control and results from this subagent to its parent.

        This method transfers results up the agent tree, validating them against
        the schema before returning. It automatically adds metadata like agent name
        and evaluation results.

        Args:
            value: Dictionary containing the task results
        """
        value["name"] = self.name
        value["description"] = self.task.description
        value["children"] = self.children
        self.schema["name"] = "string"
        self.schema["description"] = "string"
        self.schema["children"] = "list"
        self.schema["schema"] = "object"
        for callback_name, result in self.task.run_evaluations().items():
            value[callback_name] = result
            self.schema[callback_name] = {"type": ["boolean", "integer", "float"]}
        value["schema"] = self.schema
        validate(instance=value, schema=self.schema)
        # Setting this interrupts the inference loop and signals an exit
        self.completed = value

    def add_action(self, title, callback):
        """
        Register an action to be executed in the current tick.

        The agent must call this method during the action phase to specify
        what code should be executed to modify the environment.

        Args:
            title: Human-readable name for this action
            callback: Function to execute (taking self as argument)
        """
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.current_tick.action = {
            "type": "action",
            "title": title,
            "callback": callback,
        }

    def add_observation_view(self, title, callback, tool=None):
        """
        Register a new observation view that will be called on each tick.

        Args:
            title: Human-readable name for this observation
            callback: Function that returns a string representation of state
            tool: Optional name of the tool this observation is associated with
        """
        if len(self.observation_views) > 8:
            raise ValueError(
                "Maximum of 8 observation callbacks allowed. Remove one first."
            )
        view = {
            "type": "observation",
            "title": title,
            "tool": tool,
            "callback": callback,
        }
        assert type(callback) in [types.FunctionType, types.MethodType]
        self.observation_views.append(view)

    def remove_observation_view(self, view_title):
        views = [view for view in self.observation_views if view["title"] == view_title]
        for view in views:
            if "tool" in view and view["tool"] in self.tools:
                raise ValueError(
                    f"{view_title} is associated with the {view['tool']} tool. You probably don't want to remove this."
                )
            else:
                self.observation_views.remove(view)

    def update_cache(self, key, value):
        self.cache[key] = value

    def get_cache(self, key):
        return self.cache.get(key)

    def delete_cache(self, key):
        if key in self.cache:
            del self.cache[key]

    def add_evaluation(self, title, callback):
        assert type(title) == str
        assert type(callback) == types.FunctionType
        self.current_tick.evaluations.append(
            {"type": "evaluation", "title": title, "callback": callback}
        )

    def render_context(self):
        self.context = self.tree.render_context()

    def add_block(self, block):
        block["subagent"] = self.name
        block["time_remaining"] = self.end_time - time.time()
        self.tree.add_block(block)

    def add_error_block(self, error_message):
        self.debugging = True
        self.add_block({"type": "error", "message": error_message})

    def _generate_phase_block(self, block_type, mcts_params=None):
        """Generate a block for the given phase and add it to the event stream."""
        hint = self.hints.get(block_type.replace("-", "_"), "")
        weave_params = {
            "weave_n_tokens": 256,
            "weave_budget": 72,
            "weave_round_budget": 24,
            "weave_n_expand": 16,
            "weave_beam_width": 1,
            "weave_max_lookahead": 3,
            "weave_temperature": 0.2,
        }

        if mcts_params:
            weave_params.update(mcts_params)

        with open(f"/app/eval_rubrics/{block_type}.txt") as infile:
            inference_questions = infile.read().strip().splitlines()

        try:
            rprint(
                f"Writing block #[cyan]{self.tree.current_block_index()}[/cyan] of type [cyan]{block_type}[/cyan]"
            )
            block = self.generate_block(
                block_type, self.context, inference_questions, weave_params, hint=hint
            )
            self.render_context()
            return block
        except Exception:
            tb = traceback.format_exc()
            self.add_error_block(
                f"# Failed to generate {block_type} block:\n" + f'"""{tb}"""'
            )
            self.failure_stage = block_type
            return None

    def _execute_code_with_retry(self, code, phase_name, max_retries=3):
        """
        Execute code with retry logic and error handling.

        In planning mode, this simulates execution by setting up placeholder callbacks.
        In normal mode, it actually executes the Python code with exec().

        Args:
            code: Python code string to execute
            phase_name: Name of the current phase (for error reporting)
            max_retries: Maximum number of retry attempts

        Returns:
            Boolean indicating success or failure
        """
        for i in range(max_retries):
            try:
                if self.planning:
                    setup_placeholder_callbacks(self, code)
                    return True
                else:
                    exec(code)
                    return True
            except Exception:
                tb = traceback.format_exc()
                self.add_error_block(
                    f"# {phase_name} execution failed ({i + 1}/{max_retries}):\n"
                    + f'"""{tb}"""'
                )
                self.failure_stage = phase_name

                # Try to debug if not the last attempt
                if i < max_retries - 1:
                    try:
                        self._generate_phase_block("debug")
                    except:
                        pass
        return False

    def _gather_observations(self):
        """
        Collect observation data from all registered tools.

        This calls the render() method on each observation view to gather
        the current state of the environment. Each tool can provide one or more
        observation views that show different aspects of its state.

        Returns:
            Boolean indicating whether any observations were gathered
        """
        task_reminder_body = ""
        try:
            task_reminder_body += "# Problem Map:\n"
            task_reminder_body += '"""\n' + self.tree.view_board() + '\n"""'
        except Exception:
            tb = traceback.format_exc()
            self.failure_stage = "task reminder"
            self.add_error_block(
                "# TASK REMINDERS OFFLINE DUE TO CORRUPTED DATA\n" + f'"""{tb}"""'
            )

        task_blocks = [{"type": "task-reminder", "body": task_reminder_body}]

        # Collect observations
        observation_blocks = []
        for view in self.observation_views:
            try:
                if self.planning:
                    observation = simulate_observation(self, view)
                else:
                    observation = (view["title"], view["callback"](self))
                observation_blocks.append(
                    {
                        "type": "observation",
                        "title": observation[0],
                        "body": observation[1],
                    }
                )
            except Exception:
                tb = traceback.format_exc()
                self.add_error_block(
                    f"# Observation callback '{view['title']}' failed:\n"
                    + f'"""{tb}"""'
                )

        # Add all blocks to event stream
        for block in task_blocks + observation_blocks:
            self.add_block(block)

        return len(task_blocks + observation_blocks) > 0

    def _run_action_phase(self):
        """Run the action phase with retries."""
        # Generate action block
        action_block = self._generate_phase_block("action")
        if not action_block:
            return False

        self.current_tick.action_setup = action_block

        # Set up and execute action
        if self._execute_code_with_retry(action_block["body"], "action"):
            try:
                if self.planning:
                    action_result = None
                    simulated_error = roll_for_error_block(
                        self, "# Action execution failed:\n"
                    )
                    if simulated_error:
                        self.add_error_block(simulated_error)
                        return False
                else:
                    action_result = self.current_tick.action["callback"](self)

                self.current_tick.action_result = action_result
                return True
            except Exception:
                tb = traceback.format_exc()
                self.add_error_block("# Action execution failed:\n" + f'"""{tb}"""')
                try:
                    self._generate_phase_block("debug")
                except:
                    pass

        return False

    def _run_evaluation_phase(self):
        """Run evaluation setup and callbacks."""
        # Generate evaluation block
        eval_block = self._generate_phase_block("evaluation")
        if not eval_block or not self._execute_code_with_retry(
            eval_block["body"], "evaluation"
        ):
            return False

        self.current_tick.evaluation_setup = [eval_block]
        return True

    def _collect_outcomes(self, action_result):
        """Collect all outcomes from action and evaluation callbacks."""
        # Run task evaluation callbacks
        task_evaluation_results = []
        for evaluation in self.task.evaluations:
            try:
                if self.planning:
                    result = None
                else:
                    result = evaluation["callback"](self)
                task_evaluation_results.append([evaluation["title"], result])
            except Exception:
                tb = traceback.format_exc()
                task_evaluation_results.append([evaluation["title"], "ERROR"])

        # Run action evaluation callbacks
        action_evaluation_results = []
        for evaluation in self.current_tick.evaluations:
            try:
                if self.planning:
                    result = None
                    simulated_error = roll_for_error_block(
                        self, "# Evaluation failed: \n"
                    )
                    if simulated_error:
                        raise Exception
                else:
                    result = evaluation["callback"](self)

                # Stringify complex results
                if type(result) not in [str, int, bool, float, type(None)]:
                    result = repr(result)

                action_evaluation_results.append([evaluation["title"], result])
            except Exception:
                if self.planning:
                    self.add_error_block(simulated_error)
                else:
                    tb = traceback.format_exc()
                    self.add_error_block("# Evaluation failed: \n" + f'"""{tb}"""')
                action_evaluation_results.append([evaluation["title"], "ERROR"])

        # Compile all outcomes
        outcomes = []
        try:
            if self.planning:
                outcomes += [[self.current_tick.action["title"], None]]
            else:
                outcomes += [[self.current_tick.action["title"], action_result]]
        except AttributeError:
            outcomes += [("[No action specified with agent.add_action()]", "ERROR")]

        # Add all evaluation results
        outcomes += task_evaluation_results
        outcomes += action_evaluation_results

        return outcomes

    def generate_block(
        self, block_type, context, eval_questions, weave_params, hint=""
    ):
        """Generate a block and add it to the event stream."""
        return generate_block_inner(
            self, block_type, context, eval_questions, weave_params, hint
        )

    def tick(self):
        """
        Execute a complete tick cycle through all phases.

        A tick is the fundamental unit of agent operation, consisting of:
        1. Gathering observations from the environment
        2. Orientation (reflection and planning)
        3. Action (executing code)
        4. Expectation (predicting results)
        5. Observation inference (adjusting what to observe)
        6. Evaluation (checking if actions worked)
        7. Outcome (summarizing results)

        Each phase generates a block that is added to the event stream.
        """
        try:
            if "ERROR" in [
                outcome[1] for outcome in self.current_tick.outcome["table"]
            ]:
                self.debugging = True
        except AttributeError:
            self.debugging = True

        # Initialize new tick
        self.current_tick = Tick(self, len(self.ticks))

        # PHASE 1: Gather observations and context
        if not self._gather_observations():
            return

        self.render_context()
        self.tree.dump_event_stream()

        # PHASE 2: Orientation
        orientation_block = self._generate_phase_block(
            "orientation",
            {
                "weave_n_tokens": 256,
                "weave_budget": 288,
                "weave_round_budget": 96,
                "weave_n_expand": 32,
            },
        )
        if not orientation_block:
            return
        self.current_tick.orientation = orientation_block

        # PHASE 3: Action
        if not self._run_action_phase():
            return

        # PHASE 4: Expectation
        expectation_block = self._generate_phase_block("expectation")
        if not expectation_block:
            return
        self.current_tick.expectation = expectation_block

        # PHASE 5: Observation inference
        inference_block = self._generate_phase_block("observation-inference")
        if not inference_block or not self._execute_code_with_retry(
            inference_block["body"], "observation-inference"
        ):
            return
        self.current_tick.observation_inference = inference_block

        # PHASE 6: Evaluation
        if not self._run_evaluation_phase():
            return

        # PHASE 7: Generate outcome
        outcomes = self._collect_outcomes(
            getattr(self.current_tick, "action_result", None)
        )

        # Create and add outcome block
        outcome_block = {
            "type": "outcome",
            "subagent": self.name,
            "index": self.tree.current_block_index() + 1,
            "timestamp": time.time(),
            "time_remaining": self.end_time - time.time(),
            "table": outcomes,
        }

        if self.planning:
            outcome_block = simulate_outcomes(self.model_name, outcome_block)

        self.add_block(outcome_block)
        self.current_tick.outcome = outcome_block

        # Finalize tick
        try:
            if not self.planning:
                self.current_tick.validate()
        except Exception:
            tb = traceback.format_exc()
            self.add_error_block("# Tick validation failed: \n" + f'"""{tb}"""')
            self.current_tick.valid = False

        self.ticks.append(self.current_tick)
        self.debugging = False
        self.failure_stage = "event stream"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_name", help="The model to use.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer to use (if different from model_name)",
    )
    parser.add_argument("--port", default=5000, help="The port to use for VLLM.")
    parser.add_argument(
        "--bootstrap", default="bootstrap.py", help="The filepath to run as bootstrap."
    )
    parser.add_argument(
        "--budget", type=int, default=360, help="Time budget for the run in minutes."
    )
    args = parser.parse_args()

    def simple_evaluate_outputs(score_prompt_fns, texts):
        if type(texts) == str:
            texts = [
                texts,
            ]
        if type(score_prompt_fns) in [types.FunctionType, functools.partial]:
            score_prompt_fns = [
                score_prompt_fns,
            ]
        scores = asyncio.run(
            evaluate_outputs_vllm(
                args.model_name, score_prompt_fns, texts, port=args.port
            )
        )
        return torch.sigmoid(scores)

    def simple_bayes_evaluate_outputs(parent_q, questions, texts):
        if type(texts) == str:
            texts = [
                texts,
            ]
        score_prompt_fns = [
            make_simple_bayes_score_prompt(question) for question in questions
        ]
        scores = asyncio.run(
            bayesian_evaluate_outputs_vllm(
                args.model_name, parent_q, score_prompt_fns, texts, port=args.port
            )
        )
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
    schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")
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
        genesis_block = {"type": "genesis", "body": infile.read()}
        self.add_block(genesis_block)

    with open(args.bootstrap) as infile:
        # Bootstrap block
        bootstrap_block = {"type": "bootstrap", "body": infile.read()}
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
            evaluation_results.append((evaluation["title"], result))

        outcomes = []
        outcomes += [
            (subagent.current_tick.action["title"], action_result),
        ]
        outcomes += evaluation_results

        # Add outcome block
        outcome_block = {"type": "outcome", "table": outcomes}
        subagent.add_block(outcome_block)
        subagent.current_tick.outcome = outcome_block

    run_bootstrap_callbacks(self)
    # Clean up mock bootstrap agent
    del self

    if not os.path.exists("/app/weave-agent-logs"):
        os.mkdir("/app/weave-agent-logs")

    result, event_stream = agent.run("main")

    with open(f"/app/weave-agent-logs/{round(time.time())}/log.json", "w") as outfile:
        out = {
            "model_name": args.model_name,
            "event_stream": event_stream,
            "result": result,
        }
        json.dump(out, outfile)
        outfile.flush()
