"""Dynamic slack uses online state, keeps signed urgency and reaches gradients."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from nets.dynamic_attention_model import DynamicAttentionModel
from nets.encoders.gnn_encoder import GNNEncoder
from options import get_options
from problems import PDCTRPTW, PDTRPTW
from problems.utils import nearest_neighbor_graph
from run import run_rl
from utils import load_args, load_model
from utils.data_utils import dataset_to_input
from utils.time_window_features import service_start_slack


def slack_state():
    distances = torch.zeros(2, 4, 4)
    distances[0, 0] = torch.tensor([0., 10., 5., 20.])
    distances[1, 2] = torch.tensor([7., 10., 0., 40.])
    return SimpleNamespace(
        ids=torch.tensor([[0], [1]]), prev_a=torch.tensor([[0], [2]]),
        cur_times=torch.tensor([[60.], [20.]]), distance_matrix=distances,
        window_starts=torch.tensor([[0., 50., 100., 0.], [0., 0., 40., 0.]]),
        window_ends=torch.tensor([[0., 80., 120., 65.], [0., 30., 60., 30.]]),
        speed=torch.tensor([[1.], [2.]]), time_horizon=torch.tensor([[90.], [40.]]),
        latest_end=torch.tensor([[10.], [10.]]),
    )


def make_model(problem=PDTRPTW, enabled=True, **kwargs):
    return DynamicAttentionModel(
        problem, embedding_dim=16, encoder_class=GNNEncoder, n_encode_layers=1,
        n_heads=2, normalization='layer', use_time_feature=True, use_arrival_times=True,
        scale_times=True, use_time_window_slack=enabled, **kwargs,
    )


def inputs(problem, full=False, batch_size=1):
    coords=torch.tensor([[[0.,0.],[1.,0.],[2.,0.],[0.,2.],[2.,2.]]])
    negative, matrix=nearest_neighbor_graph(coords[0].numpy(), -1 if full else 1, 'fixed')
    data=dict(all_nodes=coords, distance_matrix=torch.tensor(matrix,dtype=torch.float32)[None],
              graph=(~torch.tensor(negative,dtype=torch.bool))[None], speed=torch.tensor([[1.]]),
              time_horizon=torch.tensor([[30.]]), latest_end=torch.tensor([[10.]]),
              arrival_times=torch.tensor([[0.,0.,1.5,0.,2.5]]),
              service_times=torch.tensor([[0.,2.,0.,1.,0.]]),
              window_starts=torch.tensor([[0.,0.,3.,0.,4.]]),
              window_ends=torch.tensor([[100.,12.,16.,18.,22.]]),
              gamma=torch.ones(1,1),theta=torch.ones(1,1))
    if problem is PDCTRPTW:
        data.update(demand=torch.tensor([[0.,.65,.45,.5,.5]]), vehicle_capacity=torch.ones(1,1))
    data={k:v.repeat(batch_size,*([1]*(v.ndim-1))) for k,v in data.items()}
    if batch_size==2:
        data['arrival_times'][0].zero_()
        data['window_starts'][0].zero_()
        data['arrival_times'][1]=torch.tensor([0.,1.,5.,15.,25.])
        # Calls at 5 and 15 interrupt waiting for the first window at 20,
        # guaranteeing extra queries while the static batch row finishes.
        data['window_starts'][1]=torch.tensor([0.,20.,20.,30.,40.])
        data['window_ends'][1]=torch.tensor([100.,40.,40.,50.,60.])
    return data


@pytest.mark.parametrize("problem", [PDTRPTW, PDCTRPTW])
def test_action_lookahead_uses_only_revealed_customers_and_service_time_mean(problem):
    data = inputs(problem)
    data["arrival_times"] = torch.tensor([[0., 0., 0., 9., 10.]])
    data["window_starts"] = torch.tensor([[0., 5., 0., 0., 0.]])
    data["window_ends"] = torch.tensor([[0., 20., 3., 20., 20.]])
    data["time_horizon"] = torch.tensor([[10.]])
    data["latest_end"] = torch.tensor([[0.]])
    data["service_times_mean"] = torch.tensor([2.])
    model = make_model(problem, use_action_lookahead=True)
    state = problem.make_state(data)
    full = model._get_action_lookahead(state, state.get_mask())
    torch.testing.assert_close(full[0, 0, 1], torch.tensor([1., 1., .4, 0., .5]))
    torch.testing.assert_close(full[0, 0, 2], torch.tensor([2., 1., 0., 0., 0.]))

    changed = {key: value.clone() for key, value in data.items()}
    changed["distance_matrix"][:, 3:, :] += 70
    changed["distance_matrix"][:, :, 3:] += 70
    changed["window_ends"][:, 3:] = 1
    changed["service_times"][:] = 100
    new_state = problem.make_state(changed)
    indices = torch.tensor([0, 1, 2])
    compact = model._get_action_lookahead(new_state, new_state.get_mask()[:, :, indices], indices)
    torch.testing.assert_close(compact, full[:, :, indices])


def test_lookahead_warm_start_preserves_legacy_logits():
    data = inputs(PDTRPTW)
    control = make_model().eval()
    candidate = make_model(use_action_lookahead=True).eval()
    candidate.load_state_dict({**candidate.state_dict(), **control.state_dict()})
    state = PDTRPTW.make_state(data)
    with torch.no_grad():
        fixed = control._precompute(control._encode(data))
        expected, _ = control._get_log_p(fixed, state, state.get_mask())
        actual, _ = candidate._get_log_p(fixed, state, state.get_mask())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_slack_respects_travel_waiting_negative_values_and_each_rows_units():
    state=slack_state()
    torch.testing.assert_close(service_start_slack(state,False)[:,1:],torch.tensor([[10.,20.,-15.],[5.,20.,-10.]]))
    torch.testing.assert_close(service_start_slack(state,True)[:,1:],torch.tensor([[.1,.2,-.15],[.1,.4,-.2]]))
    # Clock advances while a node's original window stays fixed.
    state.cur_times += 10
    torch.testing.assert_close(service_start_slack(state,False)[:,1:],torch.tensor([[0.,20.,-25.],[-5.,20.,-20.]]))


def test_slack_uses_original_fixed_rows_after_state_indexing():
    state=slack_state()
    state.ids=state.ids[1:]
    state.prev_a=state.prev_a[1:]
    state.cur_times=state.cur_times[1:]
    torch.testing.assert_close(service_start_slack(state)[:,1:],torch.tensor([[.1,.4,-.2]]))


def test_slack_masks_unavailable_customers_and_aligns_compact_indices():
    state=slack_state()
    model=make_model()
    mask=torch.tensor([[[False,True,False,False]],[[False,False,True,True]]])
    features=model._get_time_window_slack(state,mask)
    torch.testing.assert_close(features[...,0],torch.tensor([[[0.,0.,.2,-.15]],[[0.,.1,0.,0.]]]))
    indices=torch.tensor([0,3,1])
    compact=model._get_time_window_slack(state,mask[:,:,indices],indices)
    torch.testing.assert_close(compact,features[:,:,indices])
    with pytest.raises(ValueError,match='decoder candidates'):
        model._get_time_window_slack(state,mask[:,:,:2])
    state.window_ends.requires_grad_()
    model._get_time_window_slack(state,mask).sum().backward()
    assert state.window_ends.grad[0,1]==0
    assert state.window_ends.grad[1,2:].eq(0).all()


@pytest.mark.parametrize('problem',[PDTRPTW,PDCTRPTW])
@pytest.mark.parametrize('full',[False,True])
@pytest.mark.parametrize('mode',['masked','recursive','recursive_plus_removal'])
def test_slack_forward_preserves_service_and_capacity_constraints(problem,full,mode):
    data=inputs(problem,full)
    model=make_model(problem,neighbors=-1 if full else 1,knn_strat='fixed').eval()
    model.set_decode_type('greedy')
    with torch.no_grad():
        cost, _, _, tour, _=model(data,mode=mode,return_pi=True,return_times=True)
    assert torch.isfinite(cost).all()
    assert sorted(n for n in tour[0].tolist() if n)==[1,2,3,4]


@pytest.mark.parametrize('problem',[PDTRPTW,PDCTRPTW])
def test_slack_backpropagates_without_reencoding_and_finished_padding_is_zero(problem):
    model=make_model(problem,neighbors=1,knn_strat='fixed').train()
    model.set_decode_type('greedy')
    calls=[]
    padding=[]
    handle=model.embedder.register_forward_hook(lambda *args:calls.append(True))
    original=model._get_log_p

    def inspect(fixed,state,mask=None,**kwargs):
        result=original(fixed,state,mask,**kwargs)
        finished=state.visited_.all(dim=-1).squeeze(1)
        if finished.any():
            padding.append(True)
            assert model._get_time_window_slack(state,mask)[finished].eq(0).all()
            assert result[0][finished,0,0].eq(0).all()
        return result

    model._get_log_p=inspect
    cost,ll,_=model(inputs(problem,batch_size=2),mode='masked')
    (cost.detach()*ll).mean().backward()
    handle.remove()
    assert calls==[True] and padding
    for weight in (model.project_time_window_slack.weight,model.node_embed.weight):
        assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum()>0


def test_knn_does_not_mask_distant_legal_decoder_actions():
    data=inputs(PDTRPTW)
    data['arrival_times'].zero_()
    model=make_model(neighbors=1,knn_strat='fixed').eval()
    state=PDTRPTW.make_state(data)
    assert not data['graph'][0,0,4] and not model.mask_graph
    encoded=model.embedder(model._init_embed(data),model.edge_embed(data['graph'].long()),data['graph'])
    logp,mask=model._get_log_p(model._precompute(encoded),state,state.get_mask())
    assert not mask[0,0,4] and logp[0,0,4].exp()>0
    complete=inputs(PDTRPTW,full=True)['graph'][0]
    assert complete.sum()==20 and not complete.diag().any()


@pytest.mark.parametrize('problem',[PDTRPTW,PDCTRPTW])
def test_topology_ablation_preserves_all_non_graph_instance_data(problem):
    settings=dict(min_customers=10,max_customers=10,min_dod=.5,max_dod=.5,
                  time_horizon_per_customer=.2,batch_size=2,num_samples=4,n_subregions=1)
    np.random.seed(19)
    local=problem.make_dataset(neighbors=.2,knn_strat='percentage',**settings)
    np.random.seed(19)
    full=problem.make_dataset(neighbors=-1,knn_strat='fixed',**settings)
    for i in range(4):
        a,b=local[i],full[i]
        for key in a:
            if key!='graph':
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        assert b['graph'].sum()==110
        assert a['graph'].sum()<b['graph'].sum()


def test_zero_slack_projection_preserves_legacy_logits_and_common_initial_weights():
    torch.manual_seed(7)
    old=make_model(enabled=False).eval()
    torch.manual_seed(7)
    new=make_model(enabled=True).eval()
    for key,value in old.state_dict().items():
        torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
    new.project_time_window_slack.weight.data.zero_()
    data=inputs(PDTRPTW)
    state=PDTRPTW.make_state(data)
    fixed=old._precompute(old.embedder(old._init_embed(data),old.edge_embed(data['graph'].long()),data['graph']))
    a,_=old._get_log_p(fixed,state,state.get_mask())
    b,_=new._get_log_p(fixed,state,state.get_mask())
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_slack_options_keep_legacy_loading_and_inherit_resume_feature(tmp_path):
    assert get_options([]).use_time_window_slack is False
    with pytest.raises(SystemExit):
        get_options(['--use-time-window-slack'])
    args=tmp_path/'args.json'
    args.write_text(json.dumps({'min_customers':10,'max_customers':10}))
    assert load_args(args)['use_time_window_slack'] is False
    args.write_text(json.dumps({'use_time_window_slack':True}))
    for flag in ('--resume','--load_path'):
        opts=get_options(['--problem','pdtrptw',flag,str(tmp_path/'epoch-0.pt')])
        assert opts.use_time_window_slack


@pytest.mark.parametrize('problem',['pdtrptw','pdctrptw'])
def test_slack_training_checkpoint_loads_and_keeps_feature(tmp_path,problem):
    opts=get_options([
        '--problem',problem,'--min-customers','6','--max-customers','6','--min_dod','.5','--max_dod','.5',
        '--validation-dods','.5','--use-time-window-slack','--use_time_feature','--scale_times','--use_arrival_times',
        '--time-horizon-per-customer','.2','--training-data-seed','0','--n_epochs','2',
        '--epoch_size','4','--batch_size','2','--rollout_size','4','--rollout_batch_size','2',
        '--val_size','2','--val_batch_size','1','--save-best-validation','--checkpoint_epochs','1',
        '--embedding_dim','16','--hidden_dim','16','--n_heads','2','--n_encode_layers','1',
        '--model','dattention','--gated','--learn_norm','--normalization','batch',
        '--neighbors','-1','--knn_strat','fixed','--no_cuda','--no_wandb','--no_tensorboard',
        '--no_videos','--no_progress_bar','--output_dir',str(tmp_path),
    ])
    run_rl(opts)
    model,args=load_model(str(Path(opts.save_dir)/'best-validation.pth'))
    assert args['use_time_window_slack'] and model.use_time_window_slack
    assert torch.isfinite(model.project_time_window_slack.weight).all()
    model.eval()
    model.set_decode_type('greedy')
    problem_type=PDTRPTW if problem=='pdtrptw' else PDCTRPTW
    with torch.no_grad():
        cost,_,_=model(inputs(problem_type,full=True),mode='recursive')
    assert torch.isfinite(cost).all()
