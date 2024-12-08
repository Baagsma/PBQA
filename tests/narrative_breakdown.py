import json
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List

from pydantic import BaseModel

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(os.path.dirname(SCRIPT_DIR))
from PBQA import DB, LLM  # run with python -m tests.narrative_breakdown

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class PartyCategory(Enum):
    CHARACTER = "character"
    FACTION = "faction"


class Party(BaseModel):
    name: str
    category: PartyCategory


class Relationship(BaseModel):
    party_one: Party
    party_two: Party
    relationship: str


class StoryBeat(BaseModel):
    pivotal_moment: str
    relevant_parties: List[Party]


class Breakdown(BaseModel):
    characters: List[Party]
    factions: List[Party]
    relationships: List[Relationship]
    story_beats: List[StoryBeat]


system_prompt = "You are a narrative breakdown tool. You break down a story into its characters, factions, relationships, and story beats. Characters are people, factions are groups of people, relationships are between two parties, and story beats are pivotal moments in the story. Reply with a json object that contains the characters, factions, relationships, and story beats of the story. Disregard all information from previous breakdowns. Focus only on the most recently provided information. The detail of your breakdown should reflect the length of the provided prose."

db = DB(host="localhost", port=6333, reset=True)
db.load_pattern(
    schema=Breakdown,
    examples="tests/breakdown.yaml",
    system_prompt=system_prompt,
    input_key="story",
)

llm = LLM(db=db, host="localhost")
llm.connect_model(
    model="llama",
    port=8080,
    stop=["<|eot_id|>", "<|start_header_id|>", "<|im_end|>"],
    temperature=0,
)

breakdown = llm.ask(
    input="""The novel opens with an introduction describing the human race as a primitive and deeply unhappy species, while also introducing an electronic encyclopedia called the *Hitchhiker's Guide to the Galaxy* which provides information on every planet in the galaxy. Earthman and Englishman Arthur Dent awakens in his home in the West Country to discover that the local planning council is trying to demolish his house to build a bypass, and lies down in front of the bulldozer to stop it. His friend Ford Prefect "wikilink") convinces the lead bureaucrat to lie down in Arthur's stead so that he can take Arthur to the local pub. The construction crew begin demolishing the house anyway, but are interrupted by the sudden arrival of a fleet of spaceships. The Vogons, the callous race of civil servants running the fleet, announce that they have come to demolish Earth to make way for a hyperspace expressway, and promptly destroy the planet. Ford and Arthur survive by hitching a ride on the spaceship, much to Arthur's amazement. Ford reveals to Arthur he is an alien researcher for the *Hitchhiker's Guide to the Galaxy*, from a small planet in the vicinity of Betelgeuse who has been posing as an out-of-work actor from Guildford for 15 years, and this was why they were able to hitch a ride on the alien ship. They are quickly discovered by the Vogons, who torture them by forcing them to listen to their poetry and then toss them out of an airlock.  Meanwhile, Zaphod Beeblebrox, Ford's "semi-cousin" and the President of the Galaxy, steals the spaceship *Heart of Gold* at its unveiling with his human companion, Trillian "wikilink"). The *Heart of Gold* is equipped with an "Infinite Improbability Drive" that allows it to travel instantaneously to any point in space by simultaneously passing through every point in the universe at once. However, the Infinite Improbability Drive has a side effect of causing impossible coincidences to occur in the physical universe. One of these improbable events occurs when Arthur and Ford are rescued by the *Heart of Gold* as it travels using the Infinite Improbability Drive. Zaphod takes his passengers --- Arthur, Ford, a depressed robot named Marvin, and Trillian --- to a legendary planet named Magrathea. Its inhabitants were said to have specialized in custom-building planets for others and to have vanished after becoming so rich that the rest of the galaxy became poor. Although Ford initially doubts that the planet is Magrathea, the planet's computers send them warning messages to leave before firing two nuclear missiles at the *Heart of Gold*. Arthur inadvertently saves them by activating the Infinite Improbability Drive improperly, which also opens an underground passage. As the ship lands, Trillian's pet mice Frankie and Benjy escape.  On Magrathea, Zaphod, Ford, and Trillian venture down to the planet's interior while leaving Arthur and Marvin outside. In the tunnels, Zaphod reveals that his actions are not a result of his own decisions, but instead motivated by neural programming that he was seemingly involved in but has no memory of. As Zaphod explains how he discovered this, the trio are trapped and knocked out with sleeping gas. On the surface, Arthur is met by a resident of Magrathea, a man named Slartibartfast, who explains that the Magratheans have been in stasis to wait out an economic recession. They have temporarily reawakened to reconstruct a second version of Earth commissioned by mice, who were in fact the most intelligent species on Earth. Slartibartfast brings Arthur to Magrathea's planet construction facility, and shows Arthur that in the distant past, a race of "hyperintelligent, pan-dimensional beings" created a supercomputer named Deep Thought to determine the answer to the "Ultimate Question to Life, the Universe, and Everything." Deep Thought eventually found the answer to be 42, an answer that made no sense because the Ultimate Question itself was not known. Because determining the Ultimate Question was too difficult even for Deep Thought, an even more advanced supercomputer was constructed for this purpose. This computer was the planet Earth, which was constructed by the Magratheans, and was five minutes away from finishing its task and figuring out the Ultimate Question when the Vogons destroyed it. The hyperintelligent superbeings participated in the program as mice, performing experiments on humans while pretending to be experimented on.  Slartibartfast takes Arthur to see his friends, who are at a feast hosted by Trillian's pet mice. The mice reject as unnecessary the idea of building a new Earth to start the process over, deciding that Arthur's brain likely contains the Ultimate Question. They offer to buy Arthur's brain, leading to a fight when he declines. The group manages to escape when the planet's security system goes off unexpectedly, but immediately run into the culprits: police in pursuit of Zaphod. The police corner Zaphod, Arthur, Ford and Trillian, and the situation seems desperate as they are trapped behind a computer bank that is about to explode from the officers' weapons firing. However, the police officers suddenly die when their life-support systems short-circuit. Suspicious, Ford discovers on the surface that Marvin became bored and explained his view of the universe to the police officers' spaceship, causing it to commit suicide. The five leave Magrathea and decide to go to The Restaurant at the End of the Universe.""",
    pattern="breakdown",
    model="llama",
)

print(f"Breakdown:\n{json.dumps(breakdown, indent=4)}\n")

breakdown = breakdown["response"]

characters = breakdown["characters"]
factions = breakdown["factions"]
relationships = breakdown["relationships"]
story_beats = breakdown["story_beats"]

assert all(
    type(character) == dict for character in characters
), f"Expected all characters to be dicts, got {characters}"
assert all(
    character["category"] == "character" for character in characters
), f"Expected all characters to be characters, got {characters}"
assert all(
    type(faction) == dict for faction in factions
), f"Expected all factions to be dicts, got {factions}"
assert all(
    faction["category"] == "faction" for faction in factions
), f"Expected all factions to be factions, got {factions}"
assert all(
    type(relationship) == dict for relationship in relationships
), f"Expected all relationships to be dicts, got {relationships}"
assert all(
    type(story_beat) == dict for story_beat in story_beats
), f"Expected all story beats to be dicts, got {story_beats}"
assert all(
    type(story_beat["relevant_parties"]) == list for story_beat in story_beats
), f"Expected all story beats to have a list of relevant parties, got {story_beats}"
assert all(
    type(story_beat["relevant_parties"][0]) == dict for story_beat in story_beats
), f"Expected all story beats to have a list of relevant parties, got {story_beats}"

log.info("All tests passed")
db.delete_collection("breakdown")
