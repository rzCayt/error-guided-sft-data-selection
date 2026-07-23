import pytest
import torch

from eg_sft.training.lora_audit import audit_lora_gradients, audit_lora_parameters


class ToyLoraModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(4, 4)
        self.lora_A = torch.nn.Linear(4, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 4, bias=False)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_B(self.lora_A(inputs))


def test_only_adapter_parameters_are_trainable_and_receive_gradients() -> None:
    model = ToyLoraModel()
    parameter_report = audit_lora_parameters(model)
    assert parameter_report.trainable_parameters > 0
    assert parameter_report.trainable_parameters < parameter_report.total_parameters

    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    gradient_report = audit_lora_gradients(model)
    assert not gradient_report.missing_trainable_gradients
    assert not gradient_report.frozen_parameters_with_gradients


def test_unexpected_trainable_base_parameter_is_rejected() -> None:
    model = ToyLoraModel()
    model.base.weight.requires_grad = True
    with pytest.raises(AssertionError, match="non-adapter"):
        audit_lora_parameters(model)


def test_real_peft_adapter_gradient_and_save_load_round_trip(tmp_path) -> None:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=16,
        n_positions=16,
        n_ctx=16,
        vocab_size=32,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        bias="none",
    )

    torch.manual_seed(20260722)
    model = get_peft_model(GPT2LMHeadModel(config), lora_config)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    audit_lora_gradients(model)
    optimizer.step()

    for name, before in frozen_before.items():
        assert torch.equal(dict(model.named_parameters())[name], before)

    model.eval()
    with torch.no_grad():
        expected_logits = model(input_ids=input_ids).logits

    adapter_dir = tmp_path / "adapter"
    model.save_pretrained(adapter_dir)

    torch.manual_seed(20260722)
    reloaded_base = GPT2LMHeadModel(config)
    reloaded = PeftModel.from_pretrained(reloaded_base, adapter_dir)
    reloaded.eval()
    with torch.no_grad():
        actual_logits = reloaded(input_ids=input_ids).logits

    assert torch.allclose(actual_logits, expected_logits, atol=1e-6, rtol=1e-6)
