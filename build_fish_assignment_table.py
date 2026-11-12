#!/usr/bin/env python3
"""Expand SurpriseWinnerEffectEvents.csv into per-video fish-state intervals."""

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path


ROLES = ("big", "treatment", "focal", "assay")
ROLE_CODES = {"big": "B", "treatment": "T", "focal": "F", "assay": "A"}
STANDARD_LOCATIONS = {
    "big": "topRight",
    "treatment": "bottomRight",
    "focal": "bottomLeft",
    "assay": "topLeft",
}
VALID_LOCATIONS = {
    "topLeft",
    "topRight",
    "bottomLeft",
    "bottomRight",
    "Left",
    "Bottom",
    "Right",
}
FIRST_DATE = date(2026, 4, 10)
LAST_DATE = date(2026, 7, 11)
DAY_START = time(6, 0)
DAY_STOP = time(18, 0)
PIS = tuple(range(13, 25))


@dataclass(frozen=True)
class Event:
    line: int
    block: int
    day: date
    at: time
    pis: tuple
    jobs: tuple
    description: str


def parse_date(value):
    value = value.strip()
    if not value:
        return None
    if value.count("/") == 1:
        value += "/26"
    return datetime.strptime(value, "%m/%d/%y").date()


def parse_time(value):
    value = value.strip()
    if not value:
        return DAY_START
    hour, minute = (int(part) for part in value.split(":"))
    # The event sheet uses informal daytime notation: 1:30 and 2:00 are PM.
    if 1 <= hour < DAY_START.hour:
        hour += 12
    return time(hour, minute)


def parse_pis(value):
    value = value.strip().lower()
    if value == "all":
        return PIS
    if value.startswith("all, no"):
        excluded = {int(number) for number in re.findall(r"\d+", value)}
        return tuple(pi for pi in PIS if pi not in excluded)
    return (int(value),)


def parse_jobs(value):
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def read_events(path):
    events = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            event_date = parse_date(row["date"])
            if event_date is None:
                continue
            description = row["event"].strip()
            # This is redundant with the F/A swap later on the same day.
            if (
                event_date == date(2026, 4, 24)
                and description.lower() == "removed"
                and parse_jobs(row["job"]) == ("focal", "assay")
            ):
                continue
            events.append(
                Event(
                    line=line,
                    block=int(row["block"]),
                    day=event_date,
                    at=parse_time(row["time"]),
                    pis=parse_pis(row["pi"]),
                    jobs=parse_jobs(row["job"]),
                    description=description,
                )
            )

    # The source sheet currently records B18.2.0 dying on May 6 and another Pi18
    # Big fish dying on June 14, but omits the intervening replacement. That fish
    # was replaced when the block-3 Treatment fish entered on May 29. Keep this
    # temporary correction only until an explicit dated Big event is added to the
    # source CSV, so correcting the source later will not introduce the fish twice.
    correction_day = date(2026, 5, 29)
    pi18_big_event_present = any(
        event.day == correction_day and 18 in event.pis and "big" in event.jobs
        for event in events
    )
    if not pi18_big_event_present:
        treatment_introduction = next(
            event
            for event in events
            if event.day == correction_day
            and 18 in event.pis
            and event.description.strip().lower() == "treatment fish in"
        )
        events.append(
            Event(
                line=max(event.line for event in events) + 1,
                block=3,
                day=correction_day,
                at=treatment_introduction.at,
                pis=(18,),
                jobs=("big",),
                description="replaced",
            )
        )
    return sorted(events, key=lambda event: (event.day, event.at, event.line))


def block_for_day(day):
    if day < date(2026, 4, 21):
        return 1
    if day < date(2026, 5, 29):
        return 2
    if day < date(2026, 6, 24):
        return 3
    return 4


class FishStates:
    def __init__(self):
        self.ids = {pi: {role: None for role in ROLES} for pi in PIS}
        self.locations = {pi: {role: None for role in ROLES} for pi in PIS}
        self.next_index = defaultdict(int)

    def introduce(self, pi, role, block):
        key = (pi, role, block)
        index = self.next_index[key]
        self.next_index[key] += 1
        self.ids[pi][role] = f"{ROLE_CODES[role]}{pi}.{block}.{index}"
        self.locations[pi][role] = STANDARD_LOCATIONS[role]

    def remove(self, pi, role):
        self.ids[pi][role] = None
        self.locations[pi][role] = None

    def put_in_standard_location(self, pi, role):
        if self.ids[pi][role] is not None:
            self.locations[pi][role] = STANDARD_LOCATIONS[role]

    def set_location(self, pi, role, location):
        if self.ids[pi][role] is not None:
            self.locations[pi][role] = location

    def apply(self, pi, event):
        description = event.description.strip().lower()

        if description in {
            "b/t fish in",
            "b/t swap",
            "f/a fish in",
            "f/a fish swap",
            "treatment fish in",
            "f/a in",
            "moved in to ts2",
        }:
            for role in event.jobs:
                self.introduce(pi, role, event.block)
            return

        if description == "replaced":
            for role in event.jobs:
                self.introduce(pi, role, event.block)
            return

        if description in {"died", "removed", "moved into home tank"}:
            for role in event.jobs:
                self.remove(pi, role)
            return

        if description == "prefight":
            self.set_location(pi, "big", "Right")
            self.set_location(pi, "treatment", "Right")
            return

        if description in {"treatment fight", "assay fight", "fight start"}:
            pair = set(event.jobs)
            if pair == {"big", "treatment"}:
                self.set_location(pi, "big", "Right")
                self.set_location(pi, "treatment", "Right")
            elif pair == {"treatment", "focal"}:
                for role in ROLES:
                    self.put_in_standard_location(pi, role)
                self.set_location(pi, "treatment", "Bottom")
                self.set_location(pi, "focal", "Bottom")
            elif pair == {"focal", "assay"}:
                for role in ROLES:
                    self.put_in_standard_location(pi, role)
                self.set_location(pi, "focal", "Left")
                self.set_location(pi, "assay", "Left")
            else:
                raise ValueError(
                    f"Unsupported fight pair on source line {event.line}: {event.jobs}"
                )
            return

        invasion = re.fullmatch(r"(big|treatment|focal|assay) invaded (big|treatment|focal|assay)", description)
        if invasion:
            invader, destination = invasion.groups()
            location = STANDARD_LOCATIONS[destination]
            self.set_location(pi, invader, location)
            self.set_location(pi, destination, location)
            return

        if description.endswith("back in the right spot.") or description.endswith(
            "back in the right spot"
        ):
            for role in event.jobs:
                self.put_in_standard_location(pi, role)
            return

        raise ValueError(
            f"Unsupported event on source line {event.line}: {event.description!r}"
        )

    def lists_for(self, pi):
        fish_ids = [self.ids[pi][role] for role in ROLES]
        fish_locations = [self.locations[pi][role] for role in ROLES]
        return fish_ids, fish_locations


def make_row(pi, day, start, stop, states):
    fish_ids, fish_locations = states.lists_for(pi)
    return {
        "video": f"pi{pi:02d}.{day:%Y.%m.%d}.06.00.mp4",
        "pi": f"pi{pi:02d}",
        "block": block_for_day(day),
        "date": day.isoformat(),
        "timeStart": start.strftime("%H:%M"),
        "timeStop": stop.strftime("%H:%M"),
        "fishIDs": json.dumps(fish_ids, separators=(",", ":")),
        "fishLocs": json.dumps(fish_locations, separators=(",", ":")),
    }


def build_rows(events):
    events_by_day_and_pi = defaultdict(list)
    for event in events:
        for pi in event.pis:
            events_by_day_and_pi[(event.day, pi)].append(event)

    states = FishStates()
    rows = []
    day = FIRST_DATE
    while day <= LAST_DATE:
        for pi in PIS:
            cursor = DAY_START
            events_by_time = defaultdict(list)
            for event in events_by_day_and_pi[(day, pi)]:
                events_by_time[event.at].append(event)

            for event_time in sorted(events_by_time):
                if event_time < DAY_START or event_time > DAY_STOP:
                    raise ValueError(
                        f"Event outside video hours on {day}: {event_time}"
                    )
                if cursor < event_time:
                    rows.append(make_row(pi, day, cursor, event_time, states))
                for event in events_by_time[event_time]:
                    states.apply(pi, event)
                cursor = event_time

            if cursor < DAY_STOP:
                rows.append(make_row(pi, day, cursor, DAY_STOP, states))
        day += timedelta(days=1)
    return rows


def validate_rows(rows):
    intervals = defaultdict(list)
    for row in rows:
        fish_ids = json.loads(row["fishIDs"])
        fish_locations = json.loads(row["fishLocs"])
        if len(fish_ids) != 4 or len(fish_locations) != 4:
            raise ValueError("Fish ID/location lists must both contain four entries.")
        if len([fish for fish in fish_ids if fish is not None]) != len(
            {fish for fish in fish_ids if fish is not None}
        ):
            raise ValueError(f"Duplicate fish ID within row: {row}")
        for fish_id, location in zip(fish_ids, fish_locations):
            if (fish_id is None) != (location is None):
                raise ValueError(f"Fish ID/location null mismatch: {row}")
            if location is not None and location not in VALID_LOCATIONS:
                raise ValueError(f"Invalid fish location {location!r}")
        intervals[(row["video"], row["pi"], row["date"])].append(
            (row["timeStart"], row["timeStop"])
        )

    expected_videos = (LAST_DATE - FIRST_DATE).days + 1
    if len(intervals) != expected_videos * len(PIS):
        raise ValueError("The table does not contain every expected Pi/day video.")
    for key, spans in intervals.items():
        spans.sort()
        if spans[0][0] != "06:00" or spans[-1][1] != "18:00":
            raise ValueError(f"Incomplete daily coverage for {key}: {spans}")
        for first, second in zip(spans, spans[1:]):
            if first[1] != second[0]:
                raise ValueError(f"Gap or overlap for {key}: {spans}")


def main():
    parser = argparse.ArgumentParser(
        description="Build long-form fish identity/location intervals from event records."
    )
    parser.add_argument(
        "--events", default="SurpriseWinnerEffectEvents.csv", help="source event CSV"
    )
    parser.add_argument(
        "--output",
        default="SurpriseWinnerEffectVideoAssignments.csv",
        help="output long-form CSV",
    )
    args = parser.parse_args()

    events = read_events(args.events)
    rows = build_rows(events)
    validate_rows(rows)

    output = Path(args.output)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "video",
                "pi",
                "block",
                "date",
                "timeStart",
                "timeStop",
                "fishIDs",
                "fishLocs",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} intervals for {len(PIS) * 93} videos to {output}")


if __name__ == "__main__":
    main()
