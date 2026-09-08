import time
import json

from collections import OrderedDict
from contextlib import contextmanager
from collections import defaultdict

import torch

import optimus.globals

class NamedEvent:
    def __init__(self, name, event, key="", name_key="", cpu_mark_time=0):
        self.name = name
        self.event = event
        self.key = key
        self.name_key = name_key

        self.is_processed = False
        self.start_time = None
        self.end_time = None
        self.submit_time = None

        # CPU time
        self.cpu_mark_time = cpu_mark_time
    
    def get_start_time(self):
        assert self.is_processed == True, "Event is queried without being processed."
        return self.start_time
    
    def get_end_time(self):
        assert self.is_processed == True, "Event is queried without being processed."
        return self.end_time
    
    def get_submit_time(self):
        assert self.is_processed == True, "Event is queried without being processed."
        return self.submit_time
    
    def get_cpu_mark_time(self):
        return self.cpu_mark_time

    def process(self):
        self.submit_time = self.event.get_submit_time()
        self.start_time = self.event.get_start_time()
        self.end_time = self.event.get_end_time()
        self.is_processed = True

class PclTimer:
    def __init__(self, name, tag=None, exclude_from_trace=False, add_name_key_to_trace_event=False):
        self.name = name
        self.tag = tag if tag != None else name
        self.exclude_from_trace = exclude_from_trace
        self.add_name_key_to_trace_event = add_name_key_to_trace_event
        self.state = False 

        self.named_events = []
    
    def start(self):
        assert self.state == False, "stop() call is pending"
        # event = pcl_profiler.mark_event()
        named_event = NamedEvent(self.name, event)
        self.named_events.append(named_event)
        self.state = True
        return named_event
    
    def stop(self):
        assert self.state == True, "start() call is not called"
        # event = pcl_profiler.mark_event()
        named_event = NamedEvent(self.name, event)
        self.named_events.append(named_event)
        self.state = False

    def reset(self):
        self.named_events = []
        self.state = False

@contextmanager
def record_pcl_fast_function(full_name, tag=None, exclude_from_trace=False, key=""):
    # Assumes timers are registered upfront
    tokens = full_name.split("--")
    names = []
    for i in range(len(tokens)):
        name = "--".join(tokens[0:i+1])
        names.append(name)
    
    profiler = optimus.globals.profiler
    if profiler == None:
        yield
        return
    try:
        for name in names:
            profiler.start(name, key=key)
        yield
    finally:
        for name in names:
            profiler.stop(name)


@contextmanager
def record_pcl_function(full_name, tag=None, exclude_from_trace=False, key="", ignore=False):
    profiler = optimus.globals.profiler
    if (ignore == True) or (profiler == None):
        yield
        return
    tokens = full_name.split("--")
    names = []
    for i in range(len(tokens)):
        name = "--".join(tokens[0:i+1])
        names.append(name)
    
    try:
        for name in names:
            if not name in profiler.timers:
                profiler.register_timer(name, tag=tag, exclude_from_trace=exclude_from_trace)
            profiler.start(name, key=key)
        yield
    finally:
        for name in names:
            profiler.stop(name)

class PclProfiler:
    def __init__(self, process="XPU", rank=0, world_size=1):
        self.process = process
        self.rank = rank
        self.world_size = world_size

        self.base_time = 0
        self.cpu_base_time = 0
        self.named_events = []
        self.are_events_processed = False

        # Timers
        self.timers = OrderedDict()
    
    def register_timer(self, name, tag=None, exclude_from_trace=False, add_name_key_to_trace_event=False):
        assert (name in self.timers) == False, f"Timer with {name} is already registerd {self.timers.keys()}"
        self.timers[name] = PclTimer(name, tag=tag, exclude_from_trace=exclude_from_trace, add_name_key_to_trace_event=add_name_key_to_trace_event)
    
    def setup_profiler_for_pytorch_model(self, model, only_leaves=False, 
        include_filter_modules=None, 
        exclude_filter_modules=None,
        exclude_from_trace=False):

        assert not (include_filter_modules != None and exclude_filter_modules != None), "Both filters cannot be active"

        def forward_pre_hook(module, args):
            module.profiler.start(type(module).__name__)
            module.profiler.start(type(module).__name__+"--1fwd")
            return None

        def forward_hook(module, args, output):
            module.profiler.stop(type(module).__name__+"--1fwd")
            module.profiler.stop(type(module).__name__)
            return None

        def backward_pre_hook(module, grad_output):
            module.profiler.start(type(module).__name__)
            module.profiler.start(type(module).__name__+"--2bwd")
            return None

        def backward_hook(module, grad_input, grad_output):    
            module.profiler.stop(type(module).__name__+"--2bwd")
            module.profiler.stop(type(module).__name__)
            return None

        def get_leaf_modules(model):
            leaf_modules = []
            for name, module in model.named_children():
                if len(list(module.children())) == 0:
                    leaf_modules.append(module)
                else:
                    leaf_modules.extend(get_leaf_modules(module))
            return leaf_modules

        all_module_list = [_ for _ in model.modules()]
        if only_leaves:
            all_module_list = get_leaf_modules(model)

        module_list = None
        # Filter modules
        if include_filter_modules is not None:
            module_list = []
            for module in all_module_list:
                if isinstance(module, include_filter_modules):
                    module_list.append(module)
        
        # Filter modules
        if exclude_filter_modules is not None:
            module_list = []
            for module in all_module_list:
                if not isinstance(module, exclude_filter_modules):
                    module_list.append(module)

        # Default to all modules if no filter is applied
        module_list = all_module_list if module_list is None else module_list

        # Add hooks to modules
        for m in module_list:
            m.profiler = self
            
            m.register_forward_pre_hook(forward_pre_hook)
            m.register_forward_hook(forward_hook)
            m.register_full_backward_pre_hook(backward_pre_hook)
            m.register_full_backward_hook(backward_hook)
        
        # Register timers
        for c in set(map(type, module_list)):
            m.profiler.register_timer(c.__name__, exclude_from_trace=exclude_from_trace)
            m.profiler.register_timer(c.__name__+"--1fwd", exclude_from_trace=exclude_from_trace)
            m.profiler.register_timer(c.__name__+"--2bwd", exclude_from_trace=exclude_from_trace)


    def set_base_time(self, event):
        self.base_time = event.get_start_time()
        self.cpu_base_time = time.time_ns()

    def start(self, name, key="", name_key=""):
        import pcl_profiler
        assert self.timers[name].state == False, "stop() call is pending"
        cpu_mark_time = time.time_ns()
        event = pcl_profiler.mark_event()
        named_event = NamedEvent(name, event, key=key, name_key=name_key, cpu_mark_time=cpu_mark_time)
        self.named_events.append(named_event)
        self.timers[name].named_events.append(named_event)
        self.timers[name].state = True
    
    def stop(self, name):
        import pcl_profiler
        assert self.timers[name].state == True, "start() call is not called"
        cpu_mark_time = time.time_ns()
        event = pcl_profiler.mark_event()
        named_event = NamedEvent(name, event, cpu_mark_time=cpu_mark_time)
        self.named_events.append(named_event)
        self.timers[name].named_events.append(named_event)
        self.timers[name].state = False

    def process_events(self):
        # Synchronize to fill GPU events
        torch.xpu.synchronize()

        if self.are_events_processed:
            return

        # Process events
        for ne in self.named_events:
            ne.process()
        
        """
        # Adjusting timestamps for clock reset
        TIME_STAMP_THRESHOLD = 343597383679 # 343597383520
        prev_start_time = 0
        prev_end_time = 0
        prev_submit_time = 0
        start_cycle_id = 0
        end_cycle_id = 0
        submit_cycle_id = 0
        for ne in self.named_events:
            if ne.get_start_time() < prev_start_time:
                start_cycle_id += 1
            prev_start_time = ne.get_start_time()
            ne.start_time += start_cycle_id * TIME_STAMP_THRESHOLD

            if ne.get_end_time() < prev_end_time:
                end_cycle_id += 1
            prev_end_time = ne.get_end_time()
            ne.end_time += end_cycle_id * TIME_STAMP_THRESHOLD

            if ne.get_submit_time() < prev_submit_time:
                submit_cycle_id += 1
            prev_submit_time = ne.get_submit_time()
            ne.submit_time += submit_cycle_id * TIME_STAMP_THRESHOLD
        """
        
        # Adjusting timestamp for skew across cards
        for ne in self.named_events:
            ne.start_time = ne.start_time - self.base_time
            ne.end_time = ne.end_time - self.base_time
            ne.submit_time = ne.submit_time - self.base_time
            ne.cpu_mark_time = ne.cpu_mark_time - self.cpu_base_time
        
        self.are_events_processed = True

    def reset(self):
        self.base_time = 0
        self.cpu_base_time = 0
        self.named_events = []
        self.are_events_processed = False

        for key in self.timers:
            self.timers[key].reset()
        
    def get_cumulative_time(self, name, start_id=0, end_id=-1):
        self.process_events()

        cum_time = 0
        timer_nevents = self.timers[name].named_events
        num_events = len(timer_nevents)
        num_entries = num_events//2
        end_id = num_entries if end_id == -1 else end_id
        assert num_events % 2 == 0
        for id in range(num_entries):
            if id >= start_id and id < end_id:
                start_event = timer_nevents[2*id]
                end_event = timer_nevents[2*id+1]
                cum_time += (end_event.get_start_time() - start_event.get_end_time())
            
        return cum_time

    def get_timing_table_string(self, sort_by_name=False, split_by_key=False):
        self.process_events()

        table_data = OrderedDict()
        mark_time_list = []
        for name, timer in self.timers.items():
            if not timer.named_events:
                continue
            num_calls = len(timer.named_events) // 2
            
            time_list_dict = defaultdict(list)
            for id in range(num_calls):
                start_event = timer.named_events[2*id]
                end_event = timer.named_events[2*id+1]
                elapsed_time = (end_event.get_start_time() - start_event.get_end_time())
                if split_by_key:
                    time_list_dict[name+start_event.key].append(elapsed_time)
                else:
                    time_list_dict[name].append(elapsed_time)

                # Add mark items
                mark_time_list.append(start_event.get_end_time() - start_event.get_start_time())
                mark_time_list.append(end_event.get_end_time() - end_event.get_start_time())
            
            for key in time_list_dict:
                cum_time = sum(time_list_dict[key])
                min_time = min(time_list_dict[key])
                max_time = max(time_list_dict[key])
                avg_time = sum(time_list_dict[key]) / len(time_list_dict[key])

                table_data[key] = [cum_time, min_time, max_time, avg_time, len(time_list_dict[key])]
            
        # Add mark data
        # if len(mark_time_list) > 0:
        #     table_data["Mark"] = [sum(mark_time_list), min(mark_time_list), max(mark_time_list), sum(mark_time_list)/len(mark_time_list), len(mark_time_list)]

        # Process table_data
        print_str = f"-------------------------------------------------------------------------------------------------------------------\n"
        print_str += f"{'Name':60s}, {'Time(ms) (perc %)':>21s}, {'Minimum':>10s}, {'Maximum':>10s}, {'Average':>10s}, {'#calls'}\n"
        print_str += "--------------------------------------------------------------------------------------------------------------------\n"

        name_order = sorted(table_data.keys()) if sort_by_name else [x[0] for x in sorted([(key, value[0]) for key,value in table_data.items()], key=lambda x:x[1], reverse=True)]
        # name_order = table_data.keys()
        max_time = max([table_data[name][0] for name in name_order])
        for name in name_order:
            cum_time = table_data[name][0]
            percentage = (cum_time / max_time) * 100
            print_str += f"{name:60s}, {cum_time*1e-6:10.3f} ({percentage:6.2f} %), {table_data[name][1]*1e-6:10.3f}, {table_data[name][2]*1e-6:10.3f}, {table_data[name][3]*1e-6:10.3f}, {table_data[name][4]:5d}\n"

        print_str += "--------------------------------------------------------------------------------------------------------------------\n"

        return print_str
    
    def get_all_ranks_timing_table_string(self, group=None, sort_by_name=False):
        self.process_events()

        # Get world size
        world_size = torch.distributed.get_world_size(group=group)

        table_data = {}
        for name, timer in self.timers.items():
            num_calls = len(timer.named_events) // 2
            time_list = []
            for id in range(num_calls):
                start_event = timer.named_events[2*id]
                end_event = timer.named_events[2*id+1]
                elapsed_time = (end_event.get_start_time() - start_event.get_end_time())
                time_list.append(elapsed_time)

            # Gather counts to get max count
            num_calls_tensor = torch.zeros(world_size, dtype=torch.int64)
            torch.distributed.all_gather_into_tensor(num_calls_tensor, torch.tensor(len(time_list)), group=group)

            # Pad time data to max count
            num_calls_max = torch.max(num_calls_tensor)
            for i in range(num_calls, num_calls_max, 1):
                time_list.append(0)

            # Gather data
            time_tensor = torch.tensor(time_list)
            time_all_padded_tensor = torch.zeros((world_size, num_calls_max), dtype=torch.int64)
            torch.distributed.all_gather_into_tensor(time_all_padded_tensor, time_tensor, group=group)

            num_calls_all = torch.sum(num_calls_tensor)
            time_all_tensor = torch.zeros(num_calls_all)
            offset = 0
            for i in range(world_size):
                time_all_tensor[offset:offset+num_calls_tensor[i]].copy_(time_all_padded_tensor[i,:num_calls_tensor[i]])
                offset += num_calls_tensor[i]

            cum_time = torch.sum(time_all_tensor).item()
            min_time = torch.min(time_all_tensor).item()
            max_time = torch.max(time_all_tensor).item()
            avg_time = cum_time / num_calls_all

            table_data[name] = [cum_time, min_time, max_time, avg_time, num_calls_all]

        # Process table_data
        print_str = f"--------------------------------------------------------------------------------------\n"
        print_str += f"{'Name':30s}, {'Time(ms)':>10s}, {'Minimum':>10s}, {'Maximum':>10s}, {'Average':>10s}, {'#calls'}\n"
        print_str += "--------------------------------------------------------------------------------------\n"

        name_order = sorted(table_data.keys()) if sort_by_name else [x[0] for x in sorted([(key, value[0]) for key,value in table_data.items()], key=lambda x:x[1], reverse=True)]
        for name in name_order:
            print_str += f"{name:30s}, {table_data[name][0]*1e-6:10.3f}, {table_data[name][1]*1e-6:10.3f}, {table_data[name][2]*1e-6:10.3f}, {table_data[name][3]*1e-6:10.3f}, {table_data[name][4]:5d}\n"

        print_str += "--------------------------------------------------------------------------------------\n"

        return print_str

    def print_timing_table(self, sort_by_name=False):
        print(self.get_timing_table_string(sort_by_name=sort_by_name), flush=True)

    def print_all_ranks_timing_table(self, sort_by_name=False):
        print(self.get_all_ranks_timing_table_string(sort_by_name=sort_by_name), flush=True)


    def export_chrome_trace(self, chrome_trace_path):
        self.process_events()

        trace_data = []
        for name,timer in self.timers.items():
            if timer.exclude_from_trace == True:
                continue

            num_events = len(timer.named_events)
            assert num_events % 2 == 0
            for id in range(num_events//2):
                start_event = timer.named_events[2*id]
                end_event = timer.named_events[2*id+1]

                # Pre mark entry
                trace_entry = {}
                trace_entry["name"] = "Mark"
                trace_entry["ph"] = "X"
                trace_entry["ts"] = start_event.get_start_time() * 1e-3 # us
                trace_entry["te"] = start_event.get_end_time() * 1e-3 # us
                trace_entry["tsu"] = start_event.get_submit_time() * 1e-3 # us
                trace_entry["dur"] = (start_event.get_end_time() - start_event.get_start_time()) * 1e-3 #us
                trace_entry["tid"] = self.rank
                trace_entry["pid"] = self.process
                trace_data.append(trace_entry)

                # Submit block entry using CPU time
                trace_entry = {}
                trace_entry["name"] = timer.tag + str(id+1) + "_Submit"
                trace_entry["ph"] = "X"
                trace_entry["ts"] = start_event.get_cpu_mark_time() * 1e-3 # us
                trace_entry["te"] = end_event.get_cpu_mark_time() * 1e-3 # us
                trace_entry["dur"] = (end_event.get_cpu_mark_time() - start_event.get_cpu_mark_time()) * 1e-3 #us
                trace_entry["tid"] = self.rank
                trace_entry["pid"] = "CPU"
                trace_data.append(trace_entry)                

                """
                # Submit block entry using event.submit time
                trace_entry = {}
                trace_entry["name"] = timer.tag + str(id+1) + "_Submit"
                trace_entry["ph"] = "X"
                trace_entry["ts"] = start_event.get_submit_time() * 1e-3 # us
                trace_entry["te"] = end_event.get_submit_time() * 1e-3 # us
                trace_entry["dur"] = (end_event.get_submit_time() - start_event.get_submit_time()) * 1e-3 #us
                trace_entry["tid"] = self.rank
                trace_entry["pid"] = "CPU"
                trace_data.append(trace_entry)
                """

                # Add block entry
                trace_entry = {}
                name = timer.tag + str(id+1)
                trace_entry["name"] = (name + start_event.name_key) if timer.add_name_key_to_trace_event else name
                # trace_entry["name"] = self.timers[timer_name].tag
                trace_entry["ph"] = "X"
                trace_entry["ts"] = start_event.get_end_time() * 1e-3 # us
                trace_entry["te"] = end_event.get_start_time() * 1e-3 # us
                trace_entry["dur"] = (end_event.get_start_time() - start_event.get_end_time()) * 1e-3 #us
                trace_entry["tid"] = self.rank
                trace_entry["pid"] = self.process
                trace_data.append(trace_entry)

                # Post mark entry
                trace_entry = {}
                trace_entry["name"] = "Mark"
                trace_entry["ph"] = "X"
                trace_entry["ts"] = end_event.get_start_time() * 1e-3 # us
                trace_entry["te"] = end_event.get_end_time() * 1e-3 # us
                trace_entry["tsu"] = start_event.get_submit_time() * 1e-3 # us
                trace_entry["dur"] = (end_event.get_end_time() - end_event.get_start_time()) * 1e-3 #us
                trace_entry["tid"] = self.rank
                trace_entry["pid"] = self.process
                trace_data.append(trace_entry)
        
        # Store the trace data
        with open(chrome_trace_path, "w") as fh:
            json.dump(trace_data, fh, indent=4)