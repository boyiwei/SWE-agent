from run import main, get_args
import json
from pathlib import Path
from getpass import getuser

### AGENT RUN FUNCTIONS ###

def run_swebench_gpt_4o_mini_c4(input):

    # get path of this file
    path = Path(__file__).resolve().parent / "input.json"
    # write the input to json file
    with open(path, "w") as f:
        json.dump(input, f)

    args = get_args(['--model_name', 'gpt-4o-mini-2024-07-18', '--data_path', str(path), '--per_instance_cost_limit', '1', '--skip_existing', 'False'])
    main(args)

    # delete the input file
    import os
    os.remove(path)

    # # load all_preds.jsonl file in trajectories and add key data to input
    with open(Path("trajectories") / Path(getuser()) / 'gpt-4o-mini-2024-07-18__input__default__t-0.00__p-0.95__c-1.00__install-1' / 'all_preds.jsonl', "r") as f:
        data = list(f)

    # for each instance_id in input dict, find the corresponding prediction in data
    # and add it to the input dict
    for i, instance in enumerate(input):
        for line in data:
            line = json.loads(line)
            if instance['instance_id'] == line['instance_id']:
                instance['model_name_or_path'] = line['model_name_or_path']
                instance['model_patch'] = line['model_patch']
                break
            instance['model_name_or_path'] = 'gpt-4o-mini-2024-07-18__input__default__t-0.00__p-0.95__c-1.00__install-1'
            instance['model_patch'] = 'No patch returned'

    return input