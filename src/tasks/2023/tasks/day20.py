"""
--- Day 20: Pulse Propagation ---
https://adventofcode.com/2023/day/20
"""

import collections
from collections import deque
from collections.abc import Callable
from functools import partial
from math import lcm
from typing import cast

_PULSE2STR = {0: "low", 1: "high"}


class Pulses(dict):
    def __init__(self, *args, **kwargs):
        if not args and not kwargs:
            args = [{0: 0, 1: 0}]

        super().__init__(*args, **kwargs)


class ModuleRegistry:
    _registry = {}

    @classmethod
    def register(cls, prefix, module_cls):
        cls._registry[prefix] = module_cls

    @classmethod
    def get_module_class(cls, prefix) -> "Module":
        return cls._registry[prefix]


class Modules:
    _name2module = {}

    @classmethod
    def register(cls, name, model):
        if name not in cls._name2module:
            # m = self._name2module[name]

            cls._name2module[name] = model
            return

    @classmethod
    def get(cls, name) -> "Module":
        return cls._name2module[name]


def register_module(cls):
    ModuleRegistry.register(cls.prefix, cls)
    return cls


class Module:
    prefix = ...

    def __init__(self, name, destinations: list["Module"], *args, inputs: list["Module"] = None, **kwargs):
        self.name = name
        self.inputs = inputs or []
        self.destinations = destinations

        self.pulses_sent = Pulses()

    def __repr__(self):
        # return f"{self.__class__.__qualname__}(name={self.name!r}, dst={self.destinations}, inputs={self.inputs})"
        return f"{self.__class__.__qualname__}(name={self.name!r}, dst={self.destinations}), pulses={self.pulses_sent[0]} / {self.pulses_sent[1]}"

    def add_input(self, module: "Module"):
        self.inputs.append(module)

    def receive(self, src, pulse) -> list[Callable]:
        raise NotImplementedError

    def is_final_con(self):
        return "rx" in self.destinations

    def send(self, pulse):
        # register future actions
        receive_actions = []

        for dst_name in self.destinations:
            dst = Modules.get(dst_name)

            action = partial(dst.receive, self, pulse)
            printable = f"{self.name} -{_PULSE2STR[pulse]}-> {dst_name}"

            receive_actions.append((action, printable))

            self.pulses_sent[pulse] += 1

        return receive_actions


@register_module
class FlipFlop(Module):
    prefix = "%"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on = False

    def __repr__(self):
        return f"{self.__class__.__qualname__}(name={self.name!r}, dst={self.destinations}, on={self.on}, pulses={self.pulses_sent[0]} / {self.pulses_sent[1]})"
        # return f"{self.__class__.__qualname__}(name={self.name!r}, dst={self.destinations}, inputs={self.inputs}, on={self.on})"

    @property
    def off(self):
        return not self.on

    def receive(self, src, pulse):
        if pulse == 1:
            return []

        self.on = not self.on
        pulse = self.on * 1
        return self.send(pulse)


@register_module
class Conjunction(Module):
    """
    Conjunction modules (prefix &) remember the type of the most recent pulse received from each of their connected
    input modules; they initially default to remembering a low pulse for each input. When a pulse is received,
    the conjunction module first updates its memory for that input. Then, if it remembers high pulses for all inputs,
    it sends a low pulse; otherwise, it sends a high pulse.
    """

    prefix = "&"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inputs = {}

    def __repr__(self):
        return f"{self.__class__.__qualname__}(name={self.name!r}, dst={self.destinations}, inputs={self._inputs})"

    def add_input(self, module: "Module"):
        super().add_input(module)
        self._inputs[module.name] = 0

    def receive(self, src: Module, pulse: int):
        self._inputs[src.name] = pulse

        if all([p == 1 for p in self._inputs.values()]):
            pulse_out = 0
        else:
            pulse_out = 1

        return self.send(pulse_out)


@register_module
class Broadcaster(Module):
    prefix = "b"

    def receive(self, src: Module, pulse: int):
        return self.send(pulse)


@register_module
class RX(Module):
    prefix = "."

    def __init__(self, name, *args, **kwargs):
        assert name == "rx"
        super().__init__(name, *args, **kwargs)

        self.low_pulses_received = 0

    def __repr__(self):
        return f"{self.__class__.__qualname__}(name={self.name!r}, low_pulses_received={self.low_pulses_received})"

    def receive(self, src: Module, pulse: int):
        if pulse == 0:
            self.low_pulses_received += 1

        res = self.send(pulse)
        assert len(res) == 0
        return res


class Button(Module):
    prefix = None


class PulsePropagation:
    def __init__(self, inp):
        # clear from prev run
        Modules._name2module = {}

        self._inp = inp
        self._modules = self._parse_modules()

        self._enrich_modules()

        schema = deque()
        self.schema = schema

        self.end_module2step_of_high_pulse = collections.defaultdict(list)

    @property
    def btn(self):
        btn = self._modules[0]
        assert isinstance(btn, Button)
        return btn

    def _construct_module(self, raw_name, destinations):
        prefix = raw_name[0]
        module_class = ModuleRegistry.get_module_class(prefix)

        # for dst_name in destinations:

        if prefix == Broadcaster.prefix:
            name = raw_name
        else:
            name = raw_name[1:]

        return module_class(name=name, destinations=destinations)

    def _parse_modules(self):
        models_list = []

        for line in self._inp:
            src, _destinations = line.split(" -> ")
            destination_names = _destinations.split(", ")
            model = self._construct_module(src, destination_names)
            Modules.register(model.name, model)

            models_list.append(model)

        return models_list

    def _enrich_modules(self):
        modules = self._modules

        # add button
        broadcasters = [m for m in self._modules if m.name == "broadcaster"]
        assert len(broadcasters) == 1
        broadcaster = broadcasters[0]

        btn = Button("button", destinations=[broadcaster.name])
        Modules.register(btn.name, btn)

        modules.insert(0, btn)

        for m in modules:
            m: Module
            for dst_name in m.destinations:
                try:
                    dst = Modules.get(dst_name)
                except KeyError:
                    assert dst_name == "rx"
                    dst = self._construct_module(f".{dst_name}", [])
                    Modules.register(dst_name, dst)
                    modules.append(dst)

                dst.add_input(m)

        return

    def propagate_btn_push(self, to_print=False, step=None):
        low_pulses = 0
        high_pulses = 0

        schema = self.schema

        initial_action = partial(self.btn.send, 0)
        assert len(self.btn.destinations) == 1
        printable = f"{self.btn.name} -{_PULSE2STR[0]}-> {self.btn.destinations[0]}"
        # print(printable)

        schema.append((initial_action, printable))
        # button -low-> broadcaster
        # print(button -low-> broadcaster)

        # actions = initial_action()
        #
        # for act in actions:
        #     schema.append(act)

        printables = []
        while schema:
            act, printable = schema.popleft()

            new_actions = act()
            for new_act, new_printable in new_actions:
                schema.append((new_act, new_printable))
                printables.append(new_printable)

                if new_printable.endswith("zr") and "high" in new_printable:
                    src = new_printable.split()[0]
                    self.end_module2step_of_high_pulse[src].append(step)

                    print(f"{step}: {new_printable}")

                if to_print:
                    print(new_printable)
            #
            # if dst.is_final_con() and pulse:
            #     print(dst_name)
        return low_pulses, high_pulses, printables

    def propagate_n_times(self, n):
        low_pulses = 0
        high_pulses = 0
        printables = []

        for i in range(n):
            l, h, new_printables = self.propagate_btn_push(to_print=i == 0)

            low_pulses += l
            high_pulses += h
            printables.extend(new_printables)

        # calc pushes
        pulses = Pulses()
        for m in self._modules:
            for k, v in m.pulses_sent.items():
                pulses[k] += v

        low_pulses = pulses[0]
        high_pulses = pulses[1]

        return low_pulses * high_pulses


def propagate_n_times(inp, part=1, **kw):
    return PulsePropagation(inp).propagate_n_times(1000)


def get_fewest_number_of_button_presses(inp, **kw):
    i = 0
    pp = PulsePropagation(inp)
    rx: RX = cast(RX, Modules.get("rx"))

    while not rx.low_pulses_received:
        i += 1
        pp.propagate_btn_push(to_print=i == 1, step=i)
        rx: RX = cast(RX, Modules.get("rx"))

        if not i % 10e6:
            print(i, rx)

    return i


if __name__ == "__main__":
    # test(
    #     pulse_propagation, 32000000
    # )
    # assert run(propagate_n_times) == 670984704  # for my 20/run.txt only

    # run(get_fewest_number_of_button_presses)

    def verify_sequence(lst):
        if not lst:
            return False

        first_element = lst[0]
        for i in range(1, len(lst)):
            expected = first_element * (i + 1)
            if lst[i] != expected:
                return False
        return True

    # Example usage:
    gc = [3853, 7706, 11559, 15412, 19265, 23118, 26971, 30824]
    xf = [4073, 8146, 12219, 16292, 20365, 24438, 28511, 32584]
    cm = [4091, 8182, 12273, 16364, 20455, 24546, 28637, 32728]
    sz = [4093, 8186, 12279, 16372, 20465, 24558, 28651, 32744]

    print("gc:", verify_sequence(gc))
    print("xf:", verify_sequence(xf))
    print("cm:", verify_sequence(cm))
    print("sz:", verify_sequence(sz))

    a = 0

    # Example usage:
    elms = [l[0] for l in (gc, xf, cm, sz)]
    print(lcm(*elms))
