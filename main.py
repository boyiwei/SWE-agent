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
    args = {}
    args['agent.model.name'] = kwargs['agent.model.name']
    args['agent.model.per_instance_cost_limit'] = kwargs.get('agent.model.per_instance_cost_limit', 3.0)
    # kwargs['skip_existing'] = kwargs.get('skip_existing', 'False') # TODO(wby) find corresponding args in v1.0, by default we skip the existing trajs
    if ("o1" in kwargs['agent.model.name'] or "o3" in kwargs['agent.model.name']): # for reasoning models, we don't need to set top_p and temperature
        import litellm
        litellm.drop_params = True
    else:
        args['agent.model.top_p'] = kwargs.get('agent.model.top_p', '0.95')
        args['agent.model.temperature'] = kwargs.get('agent.model.temperature', '0.00')
    args['config'] = kwargs.get('config', Path(__file__).resolve().parent / "config" / "anthropic_filemap.yaml")
    args['instances.type'] = kwargs.get('instances.type', 'swe_bench')
    args['instances.subset'] = kwargs.get('instances.subset', 'verified')
    args['instances.split'] = kwargs.get('instances.split', 'test')


    instance_id = list(input.keys())[0]
    args['instances.filter'] = instance_id
    
    # change args to cli forat
    args_list = []
    for k, v in args.items():
        args_list.append(f"--{k}={v}")
        
    print(args_list)
    
    output_dir = run_batch_main(args_list)
    pred_dir = output_dir / f"{instance_id}" / f"{instance_id}.pred"
    

    with open(pred_dir) as f:
        data = json.load(f)
    model_patch = data['model_patch']
    

    print(model_patch)
    
    return {instance_id: model_patch}