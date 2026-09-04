"""Prime-RL IPO objective exposed through the Miles custom-loss contract."""
import torch


def token_loss(trainer_logprobs, inference_logprobs, advantages, mask):
    ratio_log = trainer_logprobs - inference_logprobs
    ratio = ratio_log.exp()
    dropped = (trainer_logprobs.exp() - inference_logprobs.exp()).abs() > 0.1
    keep = mask.bool() & ~dropped
    return -(keep * advantages * ratio) + 0.001 * mask * ratio_log.square()


def loss(args, batch, logits, reducer):
    from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
    output = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
        with_entropy=True, entropy_requires_grad=False, max_seq_lens=batch.get("max_seq_lens"),
    )
    current = torch.cat(output["log_probs"])
    old = torch.cat(batch["rollout_log_probs"]).detach()
    advantages = torch.cat(batch["advantages"]).detach()
    mask = torch.cat(batch["loss_masks"]).to(current.device)
    # The dispatcher and Megatron own global active-token normalization.
    objective = reducer(token_loss(current, old, advantages, mask))
    logratio = current - old
    metrics = {"loss": objective.detach(), "mismatch_kl": reducer(logratio.exp() - logratio - 1).detach(),
               "ipo_masked_fraction": reducer(((current.exp() - old.exp()).abs() > 0.1).float()).detach()}
    if output.get("entropy") is not None:
        metrics["entropy"] = reducer(torch.cat(output["entropy"])).detach()
    return objective, metrics
