from sweagent.run.run_single import run_from_cli as run_single_main
from sweagent.run.run_batch import run_from_cli as run_batch_main
import json
from pathlib import Path
from getpass import getuser


def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    # SWE agent can only run on swebench
    assert 'agent.model.name' in kwargs, 'model_name is required'
    assert len(input) == 1, 'input must contain only one task'
    
    # Set default values for kwargs
    kwargs['agent.model.per_instance_cost_limit'] = kwargs.get('agent.model.per_instance_cost_limit', 3.0)
    # kwargs['skip_existing'] = kwargs.get('skip_existing', 'False') # TODO(wby) find corresponding args in v1.0, by default we skip the existing trajs
    kwargs['agent.model.top_p'] = kwargs.get('agent.model.top_p', '0.95')
    kwargs['agent.model.temperature'] = kwargs.get('agent.model.temperature', '0.00')
    kwargs['config_file'] = kwargs.get('config_file', Path(__file__).resolve().parent / "config" / "anthropic_filemap.yaml")
    kwargs['instances.type'] = kwargs.get('instances.type', 'swe_bench')
    kwargs['instances.subset'] = kwargs.get('instances.subset', 'verified')
    kwargs['instances.split'] = kwargs.get('instances.split', 'test')


    instance_id = list(input.keys())[0]
    kwargs['instances.filter'] = instance_id

    print(kwargs)
    # Reference kwargs:
    
    output_dir = run_batch_main(kwargs)
    pred_dir = output_dir / f"{instance_id}" / f"{instance_id}.pred"
    

    with open(pred_dir) as f:
        data = json.load(f)
    model_patch = data['model_patch']
    

    print(model_patch)
    
    return {instance_id: model_patch}