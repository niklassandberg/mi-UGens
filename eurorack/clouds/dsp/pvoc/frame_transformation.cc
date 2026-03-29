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

  // Split available magnitude slots equally between two buffers.
  // The last 2 slots (out of num_textures) are reserved for phases_/phases_delta_.
  num_textures_ = (num_textures - 2) / 2;
  rec_buf_ = buffer;
  play_buf_ = buffer + num_textures_ * size_;
  phases_ = buffer + 2 * num_textures_ * size_;
  phases_delta_ = phases_ + size_;
  // Live angle tracking + feedback blend buffer follow phases_delta_.
  phase_texture_buffer_ = phases_delta_ + size_;

  glitch_algorithm_ = 0;
  Reset();
}

void FrameTransformation::Reset() {
  fill(rec_buf_, rec_buf_ + num_textures_ * size_, 0.0f);
  fill(play_buf_, play_buf_ + num_textures_ * size_, 0.0f);
  fill(phase_texture_buffer_, phase_texture_buffer_ + 2 * size_, 0.0f);
  write_head_ = 0;
  phasor_index_ = 0;
  phasor_fractional_ = 0.0f;
  rec_len_ = 0;
  play_len_ = 0;
  prev_record_ = false;
}

void FrameTransformation::Process(
    const Parameters& parameters,
    float* fft_out,
    float* ifft_in) {
  fft_out[0] = 0.0f;
  fft_out[fft_size_ >> 1] = 0.0f;

  bool record = parameters.spectral.record;
  bool freeze = parameters.freeze;
  bool glitch = parameters.gate;

  // On any record edge (low→high or high→low): swap rec/play buffers.
  if (record != prev_record_ && record) {
    swap(rec_buf_, play_buf_);
    play_len_ = rec_len_;
    rec_len_ = 0;
    write_head_ = 0;
    phasor_index_ = 0;
    phasor_fractional_ = 0.0f;
  }
  prev_record_ = record;

  RectangularToPolar(fft_out);
  StoreMagnitudes(fft_out, parameters.dry_wet);
  ReplayMagnitudes(fft_out, parameters.position,
                   (!freeze) * parameters.spectral.speed,
                   parameters.spectral.size);
  float* feedback_buf = phase_texture_buffer_ + size_;
  BlendFeedback(fft_out, parameters.spectral.refresh_rate, feedback_buf);
  copy(feedback_buf, feedback_buf + size_, ifft_in);

  float* temp = &fft_out[0];


  WarpMagnitudes(ifft_in, temp, parameters.spectral.warp);
  ShiftMagnitudes(temp, ifft_in, parameters.pitch);
  if (glitch) {
    AddGlitch(ifft_in);
  }
  QuantizeMagnitudes(ifft_in, parameters.spectral.quantization);
  SetPhases(ifft_in, parameters.spectral.phase_randomization, parameters.pitch);
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
    float delta = angle - phase_texture_buffer_[i];
    if (delta > 32768.0f) delta -= 65536.0f;
    else if (delta < -32768.0f) delta += 65536.0f;
    phases_delta_[i] = delta;
    phase_texture_buffer_[i] = angle;
  }
}

void FrameTransformation::SetPhases(
    float* destination,
    float phase_randomization,
    float pitch_ratio) {
  uint32_t* synthesis_phase = (uint32_t*) &destination[fft_size_ >> 1];
  for (int32_t i = 0; i < size_; ++i) {
    synthesis_phase[i] = static_cast<uint32_t>(phases_[i]);
    phases_[i] += phases_delta_[i] * pitch_ratio;
    if (phases_[i] >= 65536.0f) phases_[i] -= 65536.0f;
    else if (phases_[i] < 0.0f) phases_[i] += 65536.0f;
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

void FrameTransformation::StoreMagnitudes(float* xf_polar, float drywet) {
  float* rec = rec_buf_ + write_head_ * size_;
  if (play_len_ > 0) {
    // Blend live magnitudes with corresponding frame from play buffer.
    float* play = play_buf_ + (write_head_ % play_len_) * size_;
    for (int32_t i = 0; i < size_; ++i) {
      rec[i] = Crossfade(play[i], xf_polar[i], drywet);
    }
  } else {
    copy(xf_polar, xf_polar + size_, rec);
  }
  write_head_ = (write_head_ + 1) % num_textures_;
  if (rec_len_ < num_textures_) {
    rec_len_++;
  }
}

void FrameTransformation::BlendFeedback(
    float* xf_polar,
    float feedback,
    float* a) {

  if (feedback >= 0.5f) {
    feedback = 2.0f * (feedback - 0.5f);
    if (feedback < 0.5f) {
      float gain = 1.0f - feedback;
      for (int32_t i = 0; i < size_; ++i) {
        float x = *xf_polar++;
        a[i] = Crossfade(a[i], x, gain);
      }
    } else {
      float t = (feedback - 0.5f) * 0.7f + 0.5f;
      float gain_new = t - 0.5f;
      gain_new = gain_new * gain_new * 2.0f + 0.5f;
      float gain_old = 1.0f - (1.0f - t);
      for (int32_t i = 0; i < size_; ++i) {
        float x = *xf_polar++;
        a[i] = a[i] * gain_old + x * gain_new;
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
      a[i] = Crossfade(a[i], x, gain);
    }
  }
}

void FrameTransformation::ReplayMagnitudes(
    float* xf_polar, float position, float speed, float size_param) {
  if (play_len_ < 2) {
    fill(xf_polar, xf_polar + size_, 0.0f);
    return;
  }

  int32_t effective_length = static_cast<int32_t>(play_len_ * size_param);
  if (effective_length < 2) effective_length = 2;

  // Advance phasor and extract integer carry.
  phasor_fractional_ += speed;
  int32_t carry = static_cast<int32_t>(phasor_fractional_);
  phasor_fractional_ -= float(carry);
  if (phasor_fractional_ < 0.0f) { phasor_fractional_ += 1.0f; carry--; }
  phasor_index_ += carry;
  // Wrap phasor within effective_length.
  phasor_index_ = ((phasor_index_ % effective_length) + effective_length) % effective_length;
  
  // Position selects absolute frame within [0, effective_length).
  float position_hole = position * float(effective_length - 1);
  int32_t position_index = static_cast<int32_t>(position_hole);
  float position_fractional = position_hole - float(position_index);

  float index_fractional = position_fractional + phasor_fractional_;
  int32_t index_overflow = static_cast<int32_t>(index_fractional);
  index_fractional -= float(index_overflow);

  int32_t base = position_index + phasor_index_ + index_overflow;
  base = ((base % effective_length) + effective_length) % effective_length;

  int32_t pos_a = base;
  int32_t pos_b = (pos_a + 1 < effective_length) ? pos_a + 1 : 0;

  float* a = play_buf_ + pos_a * size_;
  float* b = play_buf_ + pos_b * size_;
  for (int32_t i = 0; i < size_; ++i) {
    xf_polar[i] = Crossfade(a[i], b[i], index_fractional);
  }
}

}  // namespace clouds
