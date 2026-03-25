// Copyright 2014 Emilie Gillet.
//
// Author: Emilie Gillet (emilie.o.gillet@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// 
// See http://creativecommons.org/licenses/MIT/ for more information.
//
// -----------------------------------------------------------------------------
//
// Transformations applied to a single STFT slice.

#include "clouds/dsp/pvoc/frame_transformation.h"

#include <algorithm>

#include "stmlib/dsp/atan.h"
#include "stmlib/dsp/units.h"
#include "stmlib/utils/random.h"

#include "clouds/dsp/frame.h"
#include "clouds/dsp/parameters.h"

namespace clouds {

using namespace std;
using namespace stmlib;

void FrameTransformation::Init(
    float* buffer,
    int32_t fft_size,
    int32_t num_textures) {
  fft_size_ = fft_size;
  size_ = (fft_size >> 1) - kHighFrequencyTruncation;
  
  texture_buffer_ = buffer;
  num_textures_ = num_textures - 1;  // Last texture slot used for phases_/phases_delta_.
  phases_ = static_cast<uint16_t*>((void*)(&texture_buffer_[num_textures_ * size_]));
  phases_delta_ = phases_ + size_;
  // Phase ring buffer follows immediately after phases_ and phases_delta_.
  phase_texture_buffer_ = phases_delta_ + size_;

  // 7-frame working buffer follows immediately after phase ring buffer.
  fft_working_buffer_ = reinterpret_cast<float*>(phase_texture_buffer_ + num_textures_ * size_);
  for (int i = 0; i < 7; ++i) {
    fft_working_frames_[i] = fft_working_buffer_ + i * size_;
  }

  write_head_ = 0;
  glitch_algorithm_ = 0;
  Reset();
}

void FrameTransformation::Reset() {
  fill(&texture_buffer_[0], &texture_buffer_[num_textures_ * size_], 0.0f);
  fill(&phase_texture_buffer_[0], &phase_texture_buffer_[num_textures_ * size_], (uint16_t)0);
  fill(fft_working_buffer_, fft_working_buffer_ + 7 * size_, 0.0f);
  write_head_ = 0;
}

void FrameTransformation::Process(
    const Parameters& parameters,
    float* fft_out,
    float* ifft_in) {
  fft_out[0] = 0.0f;
  fft_out[fft_size_ >> 1] = 0.0f;

  bool freeze = parameters.freeze;
  bool glitch = parameters.gate;
  float pitch_ratio = SemitonesToRatio(parameters.pitch);
  
  if (!freeze) {
    RectangularToPolar(fft_out);
    StoreMagnitudes(
        fft_out,
        parameters.position,
        parameters.spectral.refresh_rate);
    // Restore phases from the replay position so synthesis uses the phases
    // of the stored audio, not the current live input. Only when not frozen —
    // during freeze phases should keep accumulating for natural-sounding sustain.
    RestorePhases(parameters.position);

  }

  float* temp = &fft_out[0];

  ReplayMagnitudes(ifft_in, parameters.position);

  // Shift working frame pointers (no data copy — just rotate the pointer array).
  float* oldest = fft_working_frames_[6];
  for (int i = 6; i > 0; --i) fft_working_frames_[i] = fft_working_frames_[i - 1];
  fft_working_frames_[0] = oldest;
  // Copy current live magnitudes into frame 0.
  copy(ifft_in, ifft_in + size_, fft_working_frames_[0]);
  // Blend feedback using the 7 working frames, runs in both freeze and non-freeze.
  BlendFeedback(ifft_in, 1.0f, parameters.spectral.refresh_rate, fft_working_frames_, 7);
  copy(fft_working_frames_[0], fft_working_frames_[0] + size_, ifft_in);

  WarpMagnitudes(ifft_in, temp, parameters.spectral.warp);
  ShiftMagnitudes(temp, ifft_in, pitch_ratio);
  if (glitch) {
    AddGlitch(ifft_in);
  }
  QuantizeMagnitudes(ifft_in, parameters.spectral.quantization);
  SetPhases(ifft_in, parameters.spectral.phase_randomization, pitch_ratio);
  PolarToRectangular(ifft_in);

  if (!glitch) {
    // Decide on which glitch algorithm will be used next time... if glitch
    // is enabled on the next frame!
    glitch_algorithm_ = stmlib::Random::GetSample() & 3;
  }

  ifft_in[0] = 0.0f;
  ifft_in[fft_size_ >> 1] = 0.0f;
}

void FrameTransformation::RectangularToPolar(float* fft_data) {
  float* real = &fft_data[0];
  float* imag = &fft_data[fft_size_ >> 1];
  float* magnitude = &fft_data[0];
  for (int32_t i = 1; i < size_; ++i) {
    uint16_t angle = fast_atan2r(imag[i], real[i], &magnitude[i]);
    phases_delta_[i] = angle;  // temporarily hold live angle for StoreMagnitudes
    // phases_[i] is the synthesis accumulator — not reset from live input
  }
}

void FrameTransformation::SetPhases(
    float* destination,
    float phase_randomization,
    float pitch_ratio) {
  uint32_t* synthesis_phase = (uint32_t*) &destination[fft_size_ >> 1];
  for (int32_t i = 0; i < size_; ++i) {
    synthesis_phase[i] = phases_[i];
    phases_[i] += static_cast<uint16_t>(
        static_cast<float>(phases_delta_[i]) * pitch_ratio);
  }
  float r = phase_randomization;
  r = (r - 0.05f) * 1.06f;
  CONSTRAIN(r, 0.0f, 1.0f);
  r *= r;
  int32_t amount = static_cast<int32_t>(r * 32768.0f);
  for (int32_t i = 0; i < size_; ++i) {
    synthesis_phase[i] += \
        static_cast<int32_t>(stmlib::Random::GetSample()) * amount >> 14;
  }
}

void FrameTransformation::PolarToRectangular(float* fft_data) {
  float* real = &fft_data[0];
  float* imag = &fft_data[fft_size_ >> 1];
  float* magnitude = &fft_data[0];
  uint32_t* angle = (uint32_t*) &fft_data[fft_size_ >> 1];
  for (int32_t i = 1; i < size_; ++i) {
    fast_p2r(magnitude[i], angle[i], &real[i], &imag[i]);
  }
  for (int32_t i = size_; i < fft_size_ >> 1; ++i) {
    real[i] = imag[i] = 0.0f;
  }
}

void FrameTransformation::AddGlitch(float* xf_polar) {
  float* x = xf_polar;
  switch (glitch_algorithm_) {
    case 0:
      // Spectral hold and blow.
      {
        // Create trails
        float held = 0.0;
        for (int32_t i = 0; i < size_; ++i) {
          if ((stmlib::Random::GetSample() & 15) == 0) {
            held = x[i];
          }
          x[i] = held;
          held = held * 1.01f;
        }
      }
      break;
      
    case 1:
      // Spectral shift up with aliasing.
      {
        float factor = 1.0f + (stmlib::Random::GetSample() & 7) / 4.0f;
        float source = 0.0f;
        for (int32_t i = 0; i < size_; ++i) {
          source += factor;
          if (source >= size_) {
            source = 0.0f;
          }
          x[i] = x[static_cast<int32_t>(source)];
        }
      }
      break;
      
    case 2:
      // Kill largest harmonic and boost second largest.
      *std::max_element(&x[0], &x[size_]) = 0.0f;
      *std::max_element(&x[0], &x[size_]) *= 8.0f;
      break;
      
    case 3:
      {
        // Nasty high-pass
        for (int32_t i = 0; i < size_; ++i) {
          uint32_t random = stmlib::Random::GetSample() & 15;
          if (random == 0) {
            x[i] *= static_cast<float>(i) / 16.0f;
          }
        }
      }
      break;
      
    default:
      break;
  }
}

void FrameTransformation::QuantizeMagnitudes(float* xf_polar, float amount) {
  if (amount <= 0.48f) {
    amount = amount * 2.0f;
    // Float STFT magnitudes are 32768x smaller than the original int16 STFT.
    // Original formula was * 0.5 / fft_size_; net int16 factor = 32768/fft_size_ = 8.
    float scale_down = 0.5f * (32768.0f / float(fft_size_)) *
        SemitonesToRatio(-108.0f * (1.0f - amount));
    float scale_up = 1.0f / scale_down;
    for (int32_t i = 0.0f; i < size_; ++i) {
      xf_polar[i] = scale_up * static_cast<float>(
          static_cast<int32_t>(scale_down * xf_polar[i]));
    }
  } else if (amount >= 0.52f) {
    amount = (amount - 0.52f) * 2.0f;
    float norm = *std::max_element(&xf_polar[0], &xf_polar[size_]);
    float inv_norm = 1.0f / (norm + 0.0001f);
    for (int32_t i = 1.0f; i < size_; ++i) {
      float x = xf_polar[i] * inv_norm;
      float warped = 4.0f * x * (1.0f - x) * (1.0f - x) * (1.0f - x);
      xf_polar[i] = (x + (warped - x) * amount) * norm;
    }
  }
}

const float kWarpPolynomials[6][4] = {
  { 10.5882f, -14.8824f, 5.29412f, 0.0f },
  { -7.3333f, +9.0, -1.79167f, 0.125f },
  { 0.0f, 0.0f, 1.0f, 0.0f },
  { 0.0f, 0.5f, 0.5f, 0.0f },
  { -7.3333f, +9.5f, -2.416667f, 0.25f },
  { -7.3333f, +9.5f, -2.416667f, 0.25f },
};

void FrameTransformation::WarpMagnitudes(
    float* source,
    float* xf_polar,
    float amount) {
  float bin_width = 1.0f / static_cast<float>(size_);
  float f = 0.0;
  
  float coefficients[4];
  amount *= 4.0f;
  MAKE_INTEGRAL_FRACTIONAL(amount);
  for (int32_t i = 0; i < 4; ++i) {
    coefficients[i] = Crossfade(
        kWarpPolynomials[amount_integral][i],
        kWarpPolynomials[amount_integral + 1][i],
        amount_fractional);
  }
  
  float a = coefficients[0];
  float b = coefficients[1];
  float c = coefficients[2];
  float d = coefficients[3];
  
  for (int32_t i = 1.0f; i < size_; ++i) {
    f += bin_width;
    float wf = (d + f * (c + f * (b + a * f))) * size_;
    xf_polar[i] = Interpolate(source, wf, 1.0f);
  }
}

void FrameTransformation::ShiftMagnitudes(
    float* source,
    float* xf_polar,
    float pitch_ratio) {
  float* destination = &xf_polar[0];
  float* temp = &xf_polar[size_];
  if (pitch_ratio == 1.0f) {
    copy(&source[0], &source[size_], &temp[0]);
  } else if (pitch_ratio > 1.0f) {
    float index = 1.0f;
    float increment = 1.0f / pitch_ratio;
    for (int32_t i = 1; i < size_; ++i) {
      temp[i] = Interpolate(source, index, 1.0f);
      index += increment;
    }
  } else {
    fill(&temp[0], &temp[size_], 0.0f);
    float index = 1.0f;
    float increment = pitch_ratio;
    for (int32_t i = 1; i < size_; ++i) {
      MAKE_INTEGRAL_FRACTIONAL(index)
      temp[index_integral] += (1.0f - index_fractional) * source[i];
      temp[index_integral + 1] += index_fractional * source[i];
      index += increment;
    }
  }
  copy(&temp[0], &temp[size_], &destination[0]);
}

void FrameTransformation::StoreMagnitudes(
    float* xf_polar,
    float position,
    float feedback) {
  float* a = &texture_buffer_[write_head_ * size_];
  copy(xf_polar, xf_polar + size_, a);
  // Store live input angles (held in phases_delta_ after RectangularToPolar)
  // so RestorePhases can derive instantaneous frequency at any replay position.
  uint16_t* pa = &phase_texture_buffer_[write_head_ * size_];
  copy(phases_delta_, phases_delta_ + size_, pa);
  write_head_ = (write_head_ + 1) % num_textures_;
}

void FrameTransformation::BlendFeedback(
    float* xf_polar,
    float position, //just be between 0 and 1... but we will call it with just 0.
    float feedback, //between 0 and 1 is guees
    float** textures,
    int32_t textures_size) {

  float index_float = position * float(textures_size - 1);
  int32_t index_int = static_cast<int32_t>(index_float);
  float index_fractional = index_float - index_int;
  float gain_a = 1.0f - index_fractional;
  float gain_b = index_fractional;
  
  float* a = textures[index_int];
  float* b = textures[index_int + (position == 1.0f ? 0 : 1)];
  
  if (feedback >= 0.5f) {
    feedback = 2.0f * (feedback - 0.5f);
    if (feedback < 0.5f) {
      gain_a *= 1.0f - feedback;
      gain_b *= 1.0f - feedback;
      for (int32_t i = 0; i < size_; ++i) {
        float x = *xf_polar++;
        a[i] = Crossfade(a[i], x, gain_a);
        b[i] = Crossfade(b[i], x, gain_b);
      }
    } else {
      float t = (feedback - 0.5f) * 0.7f + 0.5f;
      float gain_new = t - 0.5f;
      gain_new = gain_new * gain_new * 2.0f + 0.5f;
      float gain_new_a = gain_a * gain_new;
      float gain_new_b = gain_b * gain_new;
      float gain_old_a = 1.0f - gain_a * (1.0f - t);
      float gain_old_b = 1.0f - gain_b * (1.0f - t);
      for (int32_t i = 0; i < size_; ++i) {
        float x = *xf_polar++;
        a[i] = a[i] * gain_old_a + x * gain_new_a;
        b[i] = b[i] * gain_old_b + x * gain_new_b;
      }
    }
  } else {
    feedback *= 2.0f;
    feedback *= feedback;
    uint16_t threshold = feedback * 65535.0f;
    for (int32_t i = 0; i < size_; ++i) {
      float x = *xf_polar++;
      float gain = static_cast<uint16_t>(Random::GetSample()) <= threshold
          ? 1.0f : 0.0f;
      a[i] = Crossfade(a[i], x, gain_a * gain);
      b[i] = Crossfade(b[i], x, gain_b * gain);
    }
  }
}

void FrameTransformation::ReplayMagnitudes(float* xf_polar, float position) {
  float index_float = position * float(num_textures_ - 1);
  int32_t offset = static_cast<int32_t>(index_float);
  float index_fractional = index_float - static_cast<float>(offset);
  int32_t pos_a = (write_head_ - 1 - offset + 2 * num_textures_) % num_textures_;
  int32_t pos_b = (write_head_ - 2 - offset + 2 * num_textures_) % num_textures_;
  float* a = &texture_buffer_[pos_a * size_];
  float* b = &texture_buffer_[pos_b * size_];
  for (int32_t i = 0; i < size_; ++i) {
    xf_polar[i] = Crossfade(a[i], b[i], index_fractional);
  }
}

void FrameTransformation::RestorePhases(float position) {
  float index_float = position * float(num_textures_ - 1);
  int32_t offset = static_cast<int32_t>(index_float);
  float index_fractional = index_float - static_cast<float>(offset);
  int32_t pos_a = (write_head_ - 1 - offset + 2 * num_textures_) % num_textures_;
  int32_t pos_b = (write_head_ - 2 - offset + 2 * num_textures_) % num_textures_;
  uint16_t* pa = &phase_texture_buffer_[pos_a * size_];
  uint16_t* pb = &phase_texture_buffer_[pos_b * size_];
  for (int32_t i = 0; i < size_; ++i) {
    // Only update the phase advance rate from the stored frames.
    // Resetting phases_[] to the same stored value every hop causes a
    // static/robotic sound — let it accumulate freely instead.
    phases_delta_[i] = static_cast<uint16_t>(pa[i] - pb[i]);
  }
}

}  // namespace clouds
