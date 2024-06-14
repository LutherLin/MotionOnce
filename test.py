import torch

from models.naturalspeech2_pytorch import (
    EncodecWrapper,
    Model,
    NaturalSpeech2,
    SpeechPromptEncoder
)

# use encodec as an example
import pdb;pdb.set_trace()
codec = EncodecWrapper()

model = Model(
    dim = 128,
    depth = 6,
    dim_prompt = 512,
    cond_drop_prob = 0.25,                  # dropout prompt conditioning with this probability, for classifier free guidance
    condition_on_prompt = True
)

# natural speech diffusion model

diffusion = NaturalSpeech2(
    model = model,
    codec = codec,
    timesteps = 1000
)

# mock raw audio data

raw_audio = torch.randn(4, 327680)
prompt = torch.randn(4, 32768)               # they randomly excised a range on the audio for the prompt during training, eventually will take care of this auto-magically

text = torch.randint(0, 100, (4, 100))
text_lens = torch.tensor([100, 50 , 80, 100])

# forwards and backwards

loss = diffusion(
    audio = raw_audio,
    text = text,
    text_lens = text_lens,
    prompt = prompt
)

loss.backward()

# after much training

generated_audio = diffusion.sample(
    length = 1024,
    text = text,
    prompt = prompt
) # (1, 327680)