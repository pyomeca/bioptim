def constraints_to_str(l_nodes: list, phase_idx: int, node_idx: int):
    constraints_str = ""
    for count in range(len(l_nodes[phase_idx][node_idx]['constraints'])):
        target_str = "" if l_nodes[phase_idx][node_idx]['constraint_sliced_target'][count] is None else \
            f"{l_nodes[phase_idx][node_idx]['constraint_sliced_target'][count]}"
        if l_nodes[phase_idx][node_idx]['constraint_quadratic'][count]:
            constraints_str += f"{l_nodes[phase_idx][node_idx]['min_bound'][count]} ≤ "
            constraints_str += f"({l_nodes[phase_idx][node_idx]['constraints'][count]}" \
                if target_str is not "" else f"{l_nodes[phase_idx][node_idx]['constraints'][count]}"
            constraints_str += f" - {target_str})<sup>2</sup>" if target_str is not "" else ""
            constraints_str += f" ≤ {l_nodes[phase_idx][node_idx]['max_bound'][count]}<br/>" \
                               f"Parameters:<br/>"
            for param in l_nodes[phase_idx][node_idx]['constraint_params'][count]:
                constraints_str += f"{param}: " \
                                   f"{l_nodes[phase_idx][node_idx]['constraint_params'][count][f'{param}']}" \
                                   f"<br/>"
            constraints_str += f"<br/>"
        else:
            constraints_str += f"{l_nodes[phase_idx][node_idx]['min_bound'][count]} ≤ " \
                               f"{l_nodes[phase_idx][node_idx]['constraints'][count]}"
            constraints_str += f" - {target_str}" if target_str is not "" else "" \
                                                                               f" ≤ {l_nodes[phase_idx][node_idx]['max_bound'][count]}<br/>" \
                                                                               f"Parameters:<br/>"
            for param in l_nodes[phase_idx][node_idx]['constraint_params'][count]:
                constraints_str += f"{param}:" \
                                   f" {l_nodes[phase_idx][node_idx]['constraint_params'][count][f'{param}']}" \
                                   f"<br/>"
            constraints_str += f"<br/>"
    return constraints_str


def constraints_to_str(l_nodes: list, phase_idx: int, node_idx: int):
    constraints_str = ""
    for count in range(len(l_nodes[phase_idx][node_idx]['Constraints'])):
        target_str = "" if l_nodes[phase_idx][node_idx]['Constraint_sliced_target'][count] is None else \
            f"{l_nodes[phase_idx][node_idx]['Constraint_sliced_target'][count]}"
        constraints_str += f"{l_nodes[phase_idx][node_idx]['Min_bound'][count]} ≤ "
        if self.nlp[phase_idx].J[obj_idx][node_idx].quadratic:
            constraints_str += f"({l_nodes[phase_idx][node_idx]['Constraints'][count]}" if target_str is not "" else f"{l_nodes[phase_idx][node_idx]['Constraints'][count]}"
            constraints_str += f" - {target_str})<sup>2</sup>" if target_str is not "" else ""
        else:
            constraints_str += f"{l_nodes[phase_idx][node_idx]['Constraints'][count]}"
            constraints_str += f" - {target_str}" if target_str is not "" else ""
        constraints_str += f" ≤ {l_nodes[phase_idx][node_idx]['Max_bound'][count]}<br/>" \
                           f"Parameters:<br/>"
        for param in l_nodes[phase_idx][node_idx]['Constraint_params'][count]:
            constraints_str += f"{param}: " \
                               f"{l_nodes[phase_idx][node_idx]['Constraint_params'][count][f'{param}']}" \
                               f"<br/>"
        constraints_str += f"<br/>"