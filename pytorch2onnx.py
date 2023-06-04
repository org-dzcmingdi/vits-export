import torch
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


if __name__ == '__main__':
    hps = utils.get_hparams_from_file("./configs/ljs_nosdp.json")

    # stn_tst = get_text("VITS is Awesome!", hps).unsqueeze(0)
    stn_tst = torch.ones([1, 30], dtype=torch.int32)
    x_tst_lengths = torch.IntTensor([stn_tst.size(1)])
    # input_names = ["inputs", "input_lengths"]

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = utils.load_checkpoint("./logs/ascend/G_109000.pth", net_g, None)
    net_g.dec.remove_weight_norm()
    net_g.eval()


    def encoder_forward(x, x_lengths):
        return net_g.infer_encoder(x, x_lengths)


    net_g.forward = encoder_forward

    torch.onnx.export(net_g, (stn_tst, x_tst_lengths), f="./logs/ascend/vits_encoder.onnx",
                      input_names=["inputs", "inputs_length"],
                      dynamic_axes={"inputs": [1], "m_p": [2], "logs_p": [2], "w": [1]},
                      output_names=["m_p", "logs_p", "w"],
                      opset_version=11)


    def decoder_forward(m_p, logs_p, eps, y_lengths):
        return net_g.infer_decoder(m_p, logs_p, eps, y_lengths).squeeze(1)


    net_g.forward = decoder_forward
    #
    decoder_input_0 = torch.ones([1, 200, 192], dtype=torch.float)
    decoder_input_1 = torch.ones([1, 200, 192], dtype=torch.float)
    decoder_input_2 = torch.randn_like(decoder_input_0, dtype=torch.float)
    decoder_input_3 = torch.IntTensor([200])
    torch.onnx.export(net_g, (decoder_input_0, decoder_input_1, decoder_input_2, decoder_input_3),
                      f="./logs/ascend/vits_decoder.onnx",
                      input_names=["m_p", "logs_p", "eps", "y_lengths"],
                      output_names=["output"],
                      dynamic_axes={"m_p": [1], "logs_p": [1], "eps": [1], "output": [1]},
                      opset_version=11
                      )
