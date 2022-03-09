import asyncio
import websockets
import json
import time
import ntplib
from threading import Thread
from opentrons import types

metadata = {
    "protocolName": "Maestro Listener",
    "author": "Rishi Kumar",
    "source": "FRG",
    "apiLevel": "2.10",
}

mixing_netlist = []


def run(protocol_context):
    # define your hardware
    tips = {}
    labwares = {}

    tip_racks = list(tips.keys())
    pipette = protocol_context.load_instrument(
        "p300_single_gen2", "right", tip_racks=tip_racks
    )

    # run through the mixing
    for generation in mixing_netlist:
        for source_str, destination_strings in generation.items():
            source_labware, source_well = source_str.split("-")
            source = labwares[source_labware][source_well]

            destinations = []
            volumes = []
            for destination_str, volume in destination_strings.items():
                destination_labware, destination_well = destination_str.split("-")
                destinations.append(labwares[destination_labware][destination_well])
                volumes.append(volume)

            pipette.transfer(
                volume=volumes,
                source=source,
                dest=destinations,
                disposal_volume=0,
                carryover=True,
                mix_before=(3, 50),
                new_tip="once",
                blow_out=True,
            )
