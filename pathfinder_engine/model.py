import time
from abc import ABC, abstractmethod
import torch
torch.set_num_threads(1)
import pathfinder_pcpage_functions as pf_pcpage
from collections import OrderedDict

class MLPrefetchModel_pathfinder_pcpage(object):
    '''
    Pathfinder SNN Prefetcher Model with STDP learning and lookahead delta prediction.
    '''

    def __init__(self):
        self.pattern_length = 3
        self.confidence_threshold = 1
        self.min_confidence = -2
        self.delta_range_length = 127
        self.neuron_numbers = 50
        self.timestamps = 32
        self.input_intensity = 1576

        self.sanity_check_index = 3000
        self.refresh_index = 5000

        self.single_network = pf_pcpage.CreateNetwork(
            self.pattern_length, self.confidence_threshold, self.min_confidence,
            self.delta_range_length, self.neuron_numbers, self.timestamps, self.input_intensity
        )
        self.training_table = OrderedDict()
        self.label_removal_counter = 0
        self.pc_removal_counter = 0
        self.page_removal_counter = 0
        self.cur_index = 0

    def load(self, path):
        pass

    def save(self, path):
        pass

    def train(self, data):
        '''
        Train the SNN model on warmup data without issuing prefetches.
        '''
        if not data:
            return

        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            self.cur_index += 1
            pc, page, offset = load_ip, load_addr >> 12, ((load_addr >> 6) & 0x3f)

            is_correct, self.label_removal_counter = pf_pcpage.check_hit(
                self.training_table, self.single_network.prediction_table, pc, page, offset, self.label_removal_counter
            )
            self.single_network.label_unlabeled(pc, page, offset, self.training_table)

            offset_prediction = 0
            if pc in self.training_table:
                inner_dict = self.training_table[pc]
                if page in inner_dict:
                    delta_pattern = pf_pcpage.calculate_deltas(inner_dict[page], offset)
                else:
                    delta_pattern = [offset, 0, 0]
                    offset_prediction = 1
            else:
                delta_pattern = [offset, 0, 0]
                offset_prediction = 1

            output_neurons, _ = self.single_network.make_prediction(delta_pattern)
            self.pc_removal_counter, self.page_removal_counter = pf_pcpage.update_training_table(
                pc, page, offset, output_neurons, delta_pattern, self.training_table, offset_prediction,
                self.pc_removal_counter, self.page_removal_counter
            )

    def generate(self, data):
        '''
        Generate lookahead prefetch addresses for evaluation trace data.
        '''
        correct_predictions = 0
        correct_offset_predictions = 0
        correct_delta_predictions = 0
        total_predictions = 0
        total_delta_prediction = 0
        total_offset_prediction = 0

        prefetch_addresses = []

        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            if self.cur_index % self.sanity_check_index == 0 and self.cur_index != 0:
                if total_predictions > 0:
                    print(f"----- current index: {self.cur_index}, correct/total: {correct_predictions / total_predictions:.4f}")

            self.cur_index += 1
            pc, page, offset = load_ip, load_addr >> 12, ((load_addr >> 6) & 0x3f)

            is_correct, self.label_removal_counter = pf_pcpage.check_hit(
                self.training_table, self.single_network.prediction_table, pc, page, offset, self.label_removal_counter
            )
            if is_correct:
                correct_predictions += 1
                if pc in self.training_table and page in self.training_table[pc] and self.training_table[pc][page][3] == 1:
                    correct_offset_predictions += 1
                else:
                    correct_delta_predictions += 1

            self.single_network.label_unlabeled(pc, page, offset, self.training_table)

            offset_prediction = 0
            if pc in self.training_table:
                inner_dict = self.training_table[pc]
                if page in inner_dict:
                    delta_pattern = pf_pcpage.calculate_deltas(inner_dict[page], offset)
                else:
                    delta_pattern = [offset, 0, 0]
                    offset_prediction = 1
            else:
                delta_pattern = [offset, 0, 0]
                offset_prediction = 1

            output_neurons, predicted_deltas = self.single_network.make_prediction(delta_pattern)

            if predicted_deltas is not None:
                for predicted_delta in predicted_deltas:
                    if offset_prediction == 1:
                        predicted_offset = predicted_delta
                    else:
                        predicted_offset = offset + predicted_delta

                    # Only emit valid in-page prefetch that is NOT the current line
                    if 0 <= predicted_offset < 64 and predicted_offset != offset:
                        if offset_prediction == 1:
                            total_offset_prediction += 1
                        else:
                            total_delta_prediction += 1

                        total_predictions += 1
                        predicted_address = (int(page) << 12) + (int(predicted_offset) << 6)
                        prefetch_addresses.append((instr_id, predicted_address))

            self.pc_removal_counter, self.page_removal_counter = pf_pcpage.update_training_table(
                pc, page, offset, output_neurons, delta_pattern, self.training_table, offset_prediction,
                self.pc_removal_counter, self.page_removal_counter
            )

        print(f"[SNN Generator] Generated {len(prefetch_addresses)} lookahead prefetches across {len(data)} loads")
        return prefetch_addresses

# Replace this if you create your own model
Model = MLPrefetchModel_pathfinder_pcpage

