"""
The states the bot can be in, and what may follow what.

The previous bot had no states. It had nested loops returning booleans, and the same
`True` meant "the node was emptied" and "I stopped being able to see the node", which is
how it came to report taking a charge from twelve trees it had never touched. Naming the
states makes that particular lie impossible to tell: leaving GATHERING for IDLE means
something was gathered, and leaving it for RECOVERING does not.
"""

from enum import Enum


class State(Enum):
    """Where the bot is in its work."""

    # Looking around for something to gather.
    SCANNING = "scanning"

    # Something was found; asking the game whether it is real before walking anywhere.
    VERIFYING = "verifying"

    # On the way to a confirmed node, mounted or not.
    TRAVELLING = "travelling"

    # Standing on a node, taking charges out of it until it gives nothing back.
    GATHERING = "gathering"

    # Nothing in sight, covering new ground.
    ROAMING = "roaming"

    # Something is wrong and is being worked around.
    RECOVERING = "recovering"

    # Asked to finish.
    STOPPING = "stopping"


# What each state is allowed to move to. Anything else is a bug in the machine rather
# than a situation in the game, and is logged as one.
ALLOWED = {
    State.SCANNING: {State.VERIFYING, State.ROAMING, State.RECOVERING, State.STOPPING},
    State.VERIFYING: {State.TRAVELLING, State.SCANNING, State.RECOVERING,
                      State.STOPPING},
    State.TRAVELLING: {State.GATHERING, State.SCANNING, State.RECOVERING,
                       State.STOPPING},
    State.GATHERING: {State.SCANNING, State.RECOVERING, State.STOPPING},
    State.ROAMING: {State.SCANNING, State.RECOVERING, State.STOPPING},
    State.RECOVERING: {State.SCANNING, State.ROAMING, State.STOPPING},
    State.STOPPING: set(),
}
