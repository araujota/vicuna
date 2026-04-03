#pragma once

#include <cstdint>
#include <iterator>
#include <string>

namespace vicuna_base64 {

template <typename InputIt, typename OutputIt>
static OutputIt encode(InputIt begin, InputIt end, OutputIt out) {
    static constexpr char alphabet[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";

    while (begin != end) {
        std::uint8_t b0 = static_cast<std::uint8_t>(*begin++);
        *out++ = alphabet[(b0 >> 2) & 0x3f];

        if (begin == end) {
            *out++ = alphabet[(b0 & 0x3) << 4];
            *out++ = '=';
            *out++ = '=';
            break;
        }

        std::uint8_t b1 = static_cast<std::uint8_t>(*begin++);
        *out++ = alphabet[((b0 & 0x3) << 4) | ((b1 >> 4) & 0x0f)];

        if (begin == end) {
            *out++ = alphabet[(b1 & 0x0f) << 2];
            *out++ = '=';
            break;
        }

        std::uint8_t b2 = static_cast<std::uint8_t>(*begin++);
        *out++ = alphabet[((b1 & 0x0f) << 2) | ((b2 >> 6) & 0x03)];
        *out++ = alphabet[b2 & 0x3f];
    }

    return out;
}

static inline std::string encode(const std::string & value) {
    std::string result;
    result.reserve(((value.size() + 2) / 3) * 4);
    encode(value.begin(), value.end(), std::back_inserter(result));
    return result;
}

} // namespace vicuna_base64
