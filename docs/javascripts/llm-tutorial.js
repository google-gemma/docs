/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Interactive widgets for "How LLMs Work (with Gemma 4)". */
(() => {
  "use strict";

  const escapeHtml = (str) =>
    String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  const animateTicker = (el, fromVal, toVal, durationMs, formatter) => {
    if (!el) return;
    if (typeof window.requestAnimationFrame !== "function") {
      el.textContent = formatter(toVal);
      return;
    }
    const start = (typeof performance !== "undefined" ? performance.now() : Date.now());
    const step = (now) => {
      const elapsed = Math.max(0, now - start);
      const t = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      const val = fromVal + (toVal - fromVal) * eased;
      el.textContent = formatter(val);
      if (t < 1 && el.isConnected) {
        window.requestAnimationFrame(step);
      }
    };
    window.requestAnimationFrame(step);
  };

  /* =====================================================================
     1. TOKENIZER PLAYGROUND (Step 1)
     Real Gemma 4 tokenizer (google/gemma-4-E2B-it, 262,144 vocab) via
     @huggingface/tokenizers in a Web Worker + IndexedDB cache + offline fallback.
     ===================================================================== */
  const DIGIT_IDS = {
    "0": 236771,
    "1": 236770,
    "2": 236778,
    "3": 236800,
    "4": 236812,
    "5": 236810,
    "6": 236825,
    "7": 236832,
    "8": 236828,
    "9": 236819,
  };

  const CONTROL_TOKENS = {
    "<pad>": { id: 0, cat: "control", note: "Padding token" },
    "<eos>": { id: 1, cat: "control", note: "End-of-sequence token" },
    "<bos>": { id: 2, cat: "control", note: "Beginning-of-sequence token" },
    "<unk>": { id: 3, cat: "control", note: "Unknown token" },
    "<mask>": { id: 4, cat: "control", note: "Mask control token" },
    "[multimodal]": { id: 5, cat: "control", note: "Multimodal marker token" },
    "<|tool>": { id: 46, cat: "control", note: "Tool declaration open" },
    "<tool|>": { id: 47, cat: "control", note: "Tool declaration close" },
    "<|tool_call>": { id: 48, cat: "control", note: "Tool call open" },
    "<tool_call|>": { id: 49, cat: "control", note: "Tool call close" },
    "<|tool_response>": { id: 50, cat: "control", note: "Tool response open / stop token" },
    "<tool_response|>": { id: 51, cat: "control", note: "Tool response close" },
    '<|"|>': { id: 52, cat: "control", note: "Gemma 4 tool string delimiter" },
    "<|think|>": { id: 98, cat: "control", note: "Activates Gemma 4 Thinking Mode" },
    "<|channel>": { id: 100, cat: "control", note: "Opens reasoning channel block" },
    "<channel|>": { id: 101, cat: "control", note: "Closes reasoning channel block" },
    "<|turn>": { id: 105, cat: "control", note: "Opens a conversation turn" },
    "<turn|>": { id: 106, cat: "control", note: "Closes turn / EOS stop token" },
    "<|image>": { id: 255999, cat: "control", note: "Beginning-of-image token" },
    "<|audio>": { id: 256000, cat: "control", note: "Beginning-of-audio token" },
    "<|image|>": { id: 258880, cat: "control", note: "Multimodal image soft-token placeholder" },
    "<|audio|>": { id: 258881, cat: "control", note: "Multimodal audio soft-token placeholder" },
    "<image|>": { id: 258882, cat: "control", note: "End-of-image token" },
    "<audio|>": { id: 258883, cat: "control", note: "End-of-audio token" },
    "<|video|>": { id: 258884, cat: "control", note: "Multimodal video placeholder" },
  };

  const getUnusedTokenId = (idx) => {
    if (!Number.isInteger(idx) || idx < 0 || idx > 6226) return null;
    if (idx <= 39) return 6 + idx;
    if (idx <= 84) return 53 + (idx - 40);
    if (idx === 85) return 99;
    if (idx <= 88) return 102 + (idx - 86);
    if (idx <= 2967) return 256001 + (idx - 89);
    return 258885 + (idx - 2968);
  };

  const BYTE_FALLBACK_CHARS = {
    "똠": [
      { t: "<0xEB>", raw: "똠[0xEB]", id: 473, cat: "subword", verified: true, note: "UTF-8 byte 1/3 of rare syllable '똠' (U+B620 → 0xEB 0x98 0xA0)" },
      { t: "<0x98>", raw: "똠[0x98]", id: 390, cat: "subword", verified: true, note: "UTF-8 byte 2/3 of rare syllable '똠' (0x98)" },
      { t: "<0xA0>", raw: "똠[0xA0]", id: 398, cat: "subword", verified: true, note: "UTF-8 byte 3/3 of rare syllable '똠' (0xA0) — zero <unk> loss!" },
    ],
    "鬮": [
      { t: "<0xE9>", raw: "鬮[0xE9]", id: 471, cat: "subword", verified: true, note: "UTF-8 byte 1/3 of rare kanji '鬮' (U+9B2E → 0xE9 0xAC 0xAE)" },
      { t: "<0xAC>", raw: "鬮[0xAC]", id: 410, cat: "subword", verified: true, note: "UTF-8 byte 2/3 of rare kanji '鬮' (0xAC)" },
      { t: "<0xAE>", raw: "鬮[0xAE]", id: 412, cat: "subword", verified: true, note: "UTF-8 byte 3/3 of rare kanji '鬮' (0xAE) — zero <unk> loss!" },
    ],
  };

  const KNOWN_VOCAB = {
    '▁': 236743, '▁▁': 138, '▁▁▁': 139, '▁▁▁▁': 140, '\n': 107, '\t': 255968, '!': 236888, '▁!': 1717, '"': 236775, '▁"': 623,
    '#': 236865, '▁#': 997, '$': 236795, '▁$': 609, '%': 236908, '▁%': 2144, '&': 236880, '▁&': 833, "'": 236789, "▁'": 756,
    '(': 236769, '▁(': 568, ')': 236768, '▁)': 1732, '*': 236829, '▁*': 808, '+': 236862, '▁+': 900, ',': 236764, '▁,': 1031,
    '-': 236772, '▁-': 753, '.': 236761, '▁.': 783, '/': 236786, '▁/': 965, '0': 236771, '1': 236770, '2': 236778, '3': 236800,
    '4': 236812, '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819, ':': 236787, '▁:': 1017, ';': 236793, '▁;': 2895,
    '<': 236820, '▁<': 655, '=': 236784, '▁=': 578, '>': 236813, '▁>': 1890, '?': 236881, '▁?': 2360, '@': 236940, '▁@': 1392,
    'A': 236776, '▁A': 562, 'B': 236799, '▁B': 603, 'C': 236780, '▁C': 565, 'D': 236796, '▁D': 622, 'E': 236788, '▁E': 645,
    'F': 236811, '▁F': 633, 'G': 236823, '▁G': 667, 'H': 236814, '▁H': 640, 'I': 236777, '▁I': 564, 'J': 236863, '▁J': 730,
    'K': 236855, '▁K': 751, 'L': 236798, '▁L': 639, 'M': 236792, '▁M': 587, 'N': 236797, '▁N': 646, 'O': 236806, '▁O': 708,
    'P': 236791, '▁P': 593, 'Q': 236935, '▁Q': 1366, 'R': 236794, '▁R': 630, 'S': 236773, '▁S': 555, 'T': 236774, '▁T': 558,
    'U': 236836, '▁U': 749, 'V': 236847, '▁V': 782, 'W': 236824, '▁W': 649, 'X': 236917, '▁X': 1684, 'Y': 236874, '▁Y': 895,
    'Z': 236953, '▁Z': 1491, '[': 236840, '▁[': 870, '\\': 236785, '▁\\': 621, ']': 236842, '▁]': 4422, '^': 236884, '▁^': 8201,
    '_': 236779, '▁_': 2222, '`': 236929, '▁`': 2165, 'a': 236746, '▁a': 496, 'b': 236763, '▁b': 518, 'c': 236755, '▁c': 505,
    'd': 236753, '▁d': 513, 'e': 236744, '▁e': 545, 'f': 236760, '▁f': 517, 'g': 236759, '▁g': 549, 'h': 236754, '▁h': 534,
    'i': 236747, '▁i': 858, 'j': 236804, '▁j': 673, 'k': 236767, '▁k': 620, 'l': 236752, '▁l': 537, 'm': 236757, '▁m': 520,
    'n': 236749, '▁n': 538, 'o': 236748, '▁o': 512, 'p': 236758, '▁p': 510, 'q': 236809, '▁q': 3752, 'r': 236750, '▁r': 637,
    's': 236751, '▁s': 503, 't': 236745, '▁t': 494, 'u': 236756, '▁u': 559, 'v': 236766, '▁v': 566, 'w': 236765, '▁w': 515,
    'x': 236781, '▁x': 1123, 'y': 236762, '▁y': 570, 'z': 236802, '▁z': 904, '{': 236782, '▁{': 642, '|': 236909, '▁|': 1109,
    '}': 236783, '▁}': 682, '~': 236995, '▁~': 2578, 'the': 1437, '▁the': 506, 'The': 818, '▁The': 669, 'an': 508, '▁an': 614,
    'An': 2267, '▁An': 1455, 'and': 624, '▁and': 532, 'or': 504, '▁or': 653, 'but': 5503, '▁but': 840, 'if': 584, '▁if': 768,
    'If': 2859, '▁If': 1637, 'in': 495, '▁in': 528, 'In': 902, '▁In': 799, 'on': 498, '▁on': 580, 'On': 2791, '▁On': 2154,
    'at': 502, '▁at': 657, 'to': 1071, '▁to': 531, 'for': 1708, '▁for': 573, 'of': 1340, '▁of': 529, 'with': 4060, '▁with': 607,
    'by': 2003, '▁by': 684, 'from': 2543, '▁from': 699, 'up': 1048, '▁up': 872, 'about': 9869, '▁about': 1003, 'into': 15040, '▁into': 1131,
    'over': 1749, '▁over': 1024, 'after': 9900, '▁after': 1308, 'is': 511, '▁is': 563, 'are': 733, '▁are': 659, 'was': 9849, '▁was': 691,
    'were': 37051, '▁were': 964, 'be': 1553, '▁be': 577, 'been': 55431, '▁been': 1010, 'being': 16898, '▁being': 1646, 'have': 17777, '▁have': 735,
    'has': 5594, '▁has': 815, 'had': 14712, '▁had': 1053, 'do': 2320, '▁do': 776, 'does': 38917, '▁does': 1677, 'did': 31275, '▁did': 1602,
    'will': 16132, '▁will': 795, 'would': 41928, '▁would': 1093, 'can': 4881, '▁can': 740, 'could': 46052, '▁could': 1451, 'should': 16223, '▁should': 1374,
    'may': 14786, '▁may': 1149, 'might': 106088, '▁might': 2473, 'must': 23600, '▁must': 1921, 'you': 7624, '▁you': 611, 'You': 3048, '▁You': 1599,
    'he': 499, '▁he': 668, 'He': 2209, '▁He': 1293, 'she': 12053, '▁she': 1304, 'She': 5778, '▁She': 2625, 'it': 509, '▁it': 625,
    'It': 1509, '▁It': 1030, 'we': 977, '▁we': 692, 'We': 1882, '▁We': 1191, 'they': 20890, '▁they': 901, 'They': 7634, '▁They': 2195,
    'this': 1580, '▁this': 672, 'This': 2094, '▁This': 1174, 'that': 7705, '▁that': 600, 'That': 6372, '▁That': 2981, 'these': 39441, '▁these': 1239,
    'those': 57587, '▁those': 1724, 'my': 3307, '▁my': 1041, 'your': 17993, '▁your': 822, 'his': 20941, '▁his': 914, 'her': 949, '▁her': 1116,
    'its': 1258, '▁its': 1061, 'our': 700, '▁our': 1023, 'their': 42974, '▁their': 910, 'what': 14070, '▁what': 1144, 'What': 3689, '▁What': 2900,
    'which': 7650, '▁which': 837, 'who': 14625, '▁who': 1015, 'Who': 15938, '▁Who': 11063, 'where': 3350, '▁where': 1298, 'Where': 10936, '▁Where': 10603,
    'when': 14730, '▁when': 1056, 'When': 4420, '▁When': 3026, 'why': 36425, '▁why': 3217, 'Why': 11355, '▁Why': 8922, 'how': 7843, '▁how': 1217,
    'How': 3910, '▁How': 2088, 'not': 2217, '▁not': 711, 'no': 1904, '▁no': 951, 'No': 3771, '▁No': 2301, 'yes': 4443, '▁yes': 11262,
    'Yes': 10784, '▁Yes': 8438, 'all': 712, '▁all': 784, 'any': 1309, '▁any': 1027, 'some': 19256, '▁some': 1070, 'many': 34717, '▁many': 1551,
    'much': 58668, '▁much': 1623, 'more': 5576, '▁more': 919, 'most': 3374, '▁most': 1346, 'other': 1538, '▁other': 1032, 'another': 51208, '▁another': 2264,
    'such': 17887, '▁such': 1288, 'only': 8960, '▁only': 1186, 'own': 931, '▁own': 1852, 'same': 31159, '▁same': 1638, 'so': 814, '▁so': 834,
    'than': 14560, '▁than': 1082, 'too': 38574, '▁too': 2311, 'very': 1158, '▁very': 1401, 'just': 4017, '▁just': 1164, 'now': 2998, '▁now': 1492,
    'then': 5215, '▁then': 1299, 'here': 8472, '▁here': 1590, 'there': 13534, '▁there': 993, 'cat': 9307, '▁cat': 5866, 'cats': 50053, '▁cats': 22797,
    'kitten': 231775, '▁kitten': 55095, 'dog': 21029, '▁dog': 4799, 'dogs': 85520, '▁dogs': 12414, '▁puppy': 34646, 'fox': 24431, '▁fox': 37423, 'bird': 18140,
    '▁bird': 8001, 'fish': 14967, '▁fish': 5385, 'animal': 36502, '▁animal': 7626, 'animals': 118441, '▁animals': 8075, 'sat': 33858, '▁sat': 2838, 'sit': 41855,
    '▁sit': 2178, 'sitting': 165797, '▁sitting': 8116, 'mat': 2956, '▁mat': 1756, 'rug': 3941, '▁rug': 17558, 'floor': 12380, '▁floor': 6413, 'bed': 4250,
    '▁bed': 4086, 'chair': 24438, '▁chair': 9048, 'table': 5127, '▁table': 2633, 'house': 6367, '▁house': 3155, 'home': 8147, '▁home': 2033, 'because': 26315,
    '▁because': 1547, 'tired': 225135, '▁tired': 20718, 'sleep': 13856, '▁sleep': 6745, 'sleeping': 89195, '▁sleeping': 20537, 'happy': 19849, '▁happy': 5293, 'fast': 8970,
    '▁fast': 4592, 'slow': 52807, '▁slow': 5111, 'big': 7485, '▁big': 2563, 'small': 10732, '▁small': 1944, 'large': 20619, '▁large': 2455, 'tiny': 67908,
    '▁tiny': 16383, 'good': 15466, '▁good': 1535, 'better': 46511, '▁better': 2480, 'best': 9783, '▁best': 1791, 'bad': 15242, '▁bad': 4287, 'new': 1951,
    '▁new': 861, 'old': 947, '▁old': 2255, 'first': 6005, '▁first': 1171, 'last': 5455, '▁last': 1774, 'long': 5843, '▁long': 1440, 'short': 15721,
    '▁short': 2822, 'high': 11480, '▁high': 1494, 'low': 976, '▁low': 2708, 'king': 8971, '▁king': 9615, 'queen': 91024, '▁queen': 26476, 'man': 1562,
    '▁man': 880, 'men': 2438, '▁men': 1758, 'woman': 16079, '▁woman': 3875, 'women': 33643, '▁women': 3607, 'person': 8269, '▁person': 1589, 'people': 21991,
    '▁people': 1331, 'child': 10869, '▁child': 1919, 'children': 12748, '▁children': 2940, 'world': 12392, '▁world': 1902, 'World': 14447, '▁World': 4109, 'earth': 40717,
    '▁earth': 7764, 'time': 2289, '▁time': 990, 'year': 3704, '▁year': 1051, 'years': 33658, '▁years': 1518, 'day': 1496, '▁day': 1719, 'days': 14356,
    '▁days': 2668, 'way': 2677, '▁way': 1595, 'thing': 2437, '▁thing': 3210, 'things': 71467, '▁things': 2432, 'life': 9524, '▁life': 1972, 'work': 2025,
    '▁work': 981, 'number': 5640, '▁number': 1548, 'fact': 18377, '▁fact': 1707, '▁Gemma': 147224, 'emma': 12367, 'Google': 19540, '▁Google': 6475, 'AI': 12553,
    '▁AI': 12498, 'model': 4368, '▁model': 2028, 'models': 11245, '▁models': 4681, 'language': 19859, '▁language': 5192, 'languages': 70496, '▁languages': 15579, 'token': 1114,
    '▁token': 8369, 'tokens': 31846, '▁tokens': 21739, 'tokenize': 81436, '▁tokenize': 178190, 'tokenizer': 111359, '▁tokenizer': 92875, 'word': 3017, '▁word': 3658, 'words': 8992,
    '▁words': 4171, 'character': 18857, '▁character': 2872, 'characters': 71271, '▁characters': 7579, 'byte': 12247, '▁byte': 16812, 'bytes': 18222, '▁bytes': 17234, 'text': 1005,
    '▁text': 1816, 'string': 2383, '▁string': 2483, 'strings': 29139, '▁strings': 16587, 'vocabulary': 159769, '▁vocabulary': 38038, 'vocab': 68034, '▁vocab': 72061, 'embedding': 67997,
    '▁embedding': 43703, 'embeddings': 92961, '▁embeddings': 106703, 'vector': 8322, '▁vector': 3550, 'vectors': 46236, '▁vectors': 16614, 'matrix': 10098, '▁matrix': 6113, 'dimension': 54800,
    '▁dimension': 11306, 'dimensions': 63265, '▁dimensions': 13376, 'semantic': 108414, '▁semantic': 52322, 'meaning': 59603, '▁meaning': 6590, 'cosine': 140523, '▁cosine': 56329, 'similarity': 86533,
    '▁similarity': 25411, 'space': 5780, '▁space': 2557, 'cluster': 22760, '▁cluster': 13627, 'layer': 7508, '▁layer': 6352, 'layers': 25232, '▁layers': 12627, 'attention': 62462,
    '▁attention': 5700, 'self': 1508, '▁self': 1265, 'query': 3278, '▁query': 7609, 'key': 2478, '▁key': 2307, 'value': 2394, '▁value': 1550, 'values': 7558,
    '▁values': 2979, 'context': 6875, '▁context': 4403, 'window': 7414, '▁window': 4771, 'local': 6047, '▁local': 2263, 'global': 13214, '▁global': 4185, 'sliding': 147327,
    '▁sliding': 31759, 'head': 2834, '▁head': 2228, 'heads': 40039, '▁heads': 15005, 'transformer': 105298, '▁transformer': 29193, 'transformers': 167864, '▁transformers': 39687, 'neural': 139356,
    '▁neural': 22823, 'network': 20746, '▁network': 3707, 'networks': 95794, '▁networks': 12230, 'deep': 28680, '▁deep': 5268, 'learning': 26661, '▁learning': 4735, 'machine': 41213,
    '▁machine': 5464, 'artificial': 141316, '▁artificial': 16477, 'intelligence': 78306, '▁intelligence': 14020, 'sample': 13271, '▁sample': 5498, 'sampling': 82939, '▁sampling': 18756, 'probability': 60423,
    '▁probability': 9948, 'logit': 155962, 'logits': 108176, '▁logits': 208253, 'softmax': 97357, '▁softmax': 210423, 'temperature': 27495, '▁temperature': 4022, 'nucleus': 214162, '▁nucleus': 31197,
    'predict': 12346, '▁predict': 4773, 'prediction': 49611, '▁prediction': 17046, 'next': 4874, '▁next': 2148, 'generate': 19057, '▁generate': 8729, 'generation': 37223, '▁generation': 8299,
    'sequence': 25425, '▁sequence': 7501, 'stream': 6678, '▁stream': 6381, 'pass': 4373, '▁pass': 1786, 'draft': 54565, '▁draft': 12262, 'verify': 41253, '▁verify': 16807,
    'verified': 87622, '▁verified': 27662, 'open': 5265, '▁open': 1932, 'weight': 6078, '▁weight': 3825, 'weights': 38357, '▁weights': 18710, '▁multimodal': 151359, 'mult': 8606,
    '▁mult': 2231, 'imodal': 98624, 'image': 3304, '▁image': 2471, 'images': 3871, '▁images': 4876, 'audio': 20156, '▁audio': 9855, 'video': 10540, '▁video': 2948,
    'vision': 13031, '▁vision': 10040, 'phone': 5275, '▁phone': 5154, 'phones': 15102, '▁phones': 23319, 'device': 10212, '▁device': 4744, 'mobile': 23104, '▁mobile': 6551,
    'run': 4503, '▁run': 1845, 'runs': 85817, '▁runs': 8784, 'running': 24825, '▁running': 4710, '▁sprint': 29991, 'think': 36345, '▁think': 1751, 'thinking': 64980,
    '▁thinking': 6972, 'thought': 45518, '▁thought': 3305, '▁reasoning': 28507, 'channel': 12027, '▁channel': 7102, 'turn': 887, '▁turn': 2490, 'user': 2364, '▁user': 2430,
    'system': 9731, '▁system': 1458, 'assistant': 111457, '▁assistant': 16326, 'tool': 13205, '▁tool': 5904, 'call': 6639, '▁call': 2246, 'response': 6275, '▁response': 3072,
    'function': 2501, '▁function': 1292, 'code': 3970, '▁code': 3393, 'python': 6719, '▁python': 23181, 'Python': 32651, '▁Python': 17856, 'json': 3723, '▁json': 8373,
    'JSON': 12324, '▁JSON': 10434, 'def': 2063, '▁def': 1096, 'return': 2060, '▁return': 994, 'import': 1106, '▁import': 1419, 'class': 1999, '▁class': 1012,
    'while': 6858, '▁while': 1651, 'elif': 36208, '▁elif': 7758, 'else': 4454, '▁else': 1663, 'True': 4339, '▁True': 6288, 'False': 9277, '▁False': 8450,
    'None': 9336, '▁None': 5450, 'print': 1995, '▁print': 1887, 'range': 8572, '▁range': 2644, 'len': 3469, '▁len': 5980, 'str': 1714, '▁str': 1540,
    'int': 720, '▁int': 801, 'float': 8344, '▁float': 6803, 'list': 2234, '▁list': 1694, 'dict': 3074, '▁dict': 12739, 'set': 1025, '▁set': 1076,
    'fib': 73368, '▁fib': 10779, 'onacci': 78113, 'greet': 183583, '▁greet': 28197, 'hello': 23391, '▁hello': 29104, 'Hello': 9259, '▁Hello': 26352, 'Bonjour': 53406,
    '▁Bonjour': 226283, 'Tokyo': 89265, '▁Tokyo': 21038, 'Paris': 50429, '▁Paris': 9079, 'France': 31756, '▁France': 7001, 'Seoul': 158058, '▁Seoul': 50215, 'Korea': 121138,
    '▁Korea': 15400, 'weather': 19323, '▁weather': 7606, 'location': 7125, '▁location': 4563, '▁unbelievable': 66917, 'un': 602, '▁un': 723, 'bel': 8145, '▁bel': 1578,
    'ievable': 179086, 'able': 742, '▁able': 2735, 'ible': 1253, 'Ant': 14054, '▁Ant': 5307, 'idis': 33090, 'establishment': 163524, '▁establishment': 18914, 'arian': 8225,
    'ism': 1929, 'super': 12666, '▁super': 2393, 'cal': 1653, '▁cal': 1421, 'rag': 16100, '▁rag': 18839, 'il': 543, '▁il': 1998, 'istic': 3316,
    'exp': 4212, '▁exp': 1520, 'ial': 706, 'id': 547, '▁id': 1846, 'ocious': 100466, 'citation': 99697, '▁citation': 56181, 're': 500, '▁re': 544,
    'pre': 1734, '▁pre': 877, 'non': 9181, '▁non': 1908, 'under': 6460, '▁under': 1208, 'sub': 2204, '▁sub': 1159, 'inter': 2266, '▁inter': 939,
    'trans': 3849, '▁trans': 1207, 'multi': 20028, '▁multi': 5074, 'auto': 9913, '▁auto': 6282, 'co': 1364, '▁co': 1028, 'de': 893, '▁de': 569,
    'dis': 2763, '▁dis': 864, 'en': 501, '▁en': 696, 'ex': 716, '▁ex': 652, 'im': 548, '▁im': 757, 'mis': 25714, '▁mis': 3930,
    'out': 725, '▁out': 855, 'pro': 1045, '▁pro': 608, 'post': 3274, '▁post': 1868, 'anti': 5027, '▁anti': 7247, 'ing': 522, '▁ing': 4616,
    'ed': 524, '▁ed': 1511, 'er': 497, '▁er': 2087, 'ers': 616, 'est': 598, '▁est': 1009, 'ly': 586, '▁ly': 14387, 'tion': 34667,
    'ation': 567, 'ition': 804, 'ization': 1854, 'sion': 142951, 'ment': 658, 'ness': 1788, 'ity': 665, 'ty': 1363, 'al': 514, '▁al': 610,
    'ic': 525, 'ical': 800, 'ous': 806, 'ious': 1434, 'eous': 39475, 'ful': 1275, 'less': 1933, '▁less': 2344, 'ive': 705, 'ative': 1280,
    'ize': 969, 'ise': 1117, 'fy': 35411, 'ify': 1891, 'ist': 617, 'ship': 5976, 'hood': 6325, 'ward': 1476, 'wards': 3259, 'wise': 5059,
    'es': 507, '▁es': 1115, 'nt': 666, 'nd': 523, 'st': 540, '▁st': 579, 'le': 519, '▁le': 674, 'ou': 521, 've': 560,
    '▁ve': 1381, 'se': 561, '▁se': 636, 'te': 654, '▁te': 926, 'ar': 516, '▁ar': 863, 'ti': 3916, 'qu': 576, '▁qu': 690,
    'as': 527, '▁as': 618, 'ip': 760, 'article': 9692, '▁article': 4676, '):': 1473, ']:': 9414, '<=': 22159, '▁<=': 6605, '>=': 11957,
    '▁>=': 6867, '==': 1224, '▁==': 1251, '!=': 20181, '▁!=': 2843, '+=': 23927, '▁+=': 3323, '->': 1160, '▁->': 3921, '=>': 9461,
    '▁=>': 1477, '//': 715, '▁//': 973, '**': 1018, '...': 1390, '▁...': 3729, '안녕하세요': 61659, '▁안녕하세요': 83732, '는': 237170, '개': 238242,
    '▁개': 18959, '▁이상의': 183126, '언': 239766, '▁언': 64822, '어를': 120857, '지원': 168800, '▁지원': 65363, '합니다': 19773, '얌': 249170, '꿍': 249927,
    '과': 237842, '▁과': 30126, '한': 237384, '▁한': 10601, '국': 238231, '어': 237430, '加': 237410, '▁加': 50624, 'さん': 9821, '▁さん': 46985,
    'さ': 237229, 'ん': 237249, '日': 237205, '本': 237261, '語': 238582, '中': 237103, '文': 237364, '🚀': 242015, '🌍': 244906, '🎉': 242061,
    '🔥': 240785, '✨': 239794, '🤖': 246143, '🧠': 249262, '💻': 244862, 'quantum': 105053, '▁quantum': 12705, 'computing': 133580, '▁computing': 20124, 'developer': 55060,
    '▁developer': 19788,
  };

  const PRESET_TOKENIZATIONS = [
    {
      label: "Gemma & Numbers",
      text: "Gemma 4 runs on phones in 2026!",
      tokens: [
        { t: "G", id: 236823, cat: "subword", verified: true, note: "Initial capital subword piece" },
        { t: "emma", id: 12367, cat: "subword", verified: true, note: "Subword continuation completing 'Gemma'" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Space before standalone number" },
        { t: "4", id: 236812, cat: "digit", verified: true, note: "Individual digit token (Gemma splits all digits 0-9)" },
        { t: "▁runs", id: 8784, cat: "word", verified: true, note: "Whole word with leading space marker ▁ (U+2581)" },
        { t: "▁on", id: 580, cat: "word", verified: true, note: "Common preposition with leading space ▁" },
        { t: "▁phones", id: 23319, cat: "word", verified: true, note: "Whole plural noun with leading space ▁" },
        { t: "▁in", id: 528, cat: "word", verified: true, note: "High-frequency preposition with leading space ▁" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Leading space before multi-digit year" },
        { t: "2", id: 236778, cat: "digit", verified: true, note: "Single digit '2' (splitting digits helps arithmetic)" },
        { t: "0", id: 236771, cat: "digit", verified: true, note: "Single digit '0'" },
        { t: "2", id: 236778, cat: "digit", verified: true, note: "Single digit '2'" },
        { t: "6", id: 236825, cat: "digit", verified: true, note: "Single digit '6'" },
        { t: "!", id: 236888, cat: "code", verified: true, note: "Punctuation token" },
      ],
    },
    {
      label: "Everyday English",
      text: "The cat sat on the mat because it was tired.",
      tokens: [
        { t: "The", id: 818, cat: "word", verified: true, note: "Sentence-initial word (no leading space)" },
        { t: "▁cat", id: 5866, cat: "word", verified: true, note: "Whole word with leading space ▁" },
        { t: "▁sat", id: 2838, cat: "word", verified: true, note: "Whole word with leading space ▁" },
        { t: "▁on", id: 580, cat: "word", verified: true, note: "High-frequency preposition" },
        { t: "▁the", id: 506, cat: "word", verified: true, note: "One of the most frequent English tokens" },
        { t: "▁mat", id: 1756, cat: "word", verified: true, note: "Whole word with leading space ▁" },
        { t: "▁because", id: 1547, cat: "word", verified: true, note: "Single token for common 7-letter conjunction" },
        { t: "▁it", id: 625, cat: "word", verified: true, note: "Pronoun token (resolved later by Self-Attention!)" },
        { t: "▁was", id: 691, cat: "word", verified: true, note: "Common past-tense verb token" },
        { t: "▁tired", id: 20718, cat: "word", verified: true, note: "Whole adjective token" },
        { t: ".", id: 236761, cat: "code", verified: true, note: "Period punctuation token" },
      ],
    },
    {
      label: "Multilingual & Emoji",
      text: "안녕하세요! Gemma 4는 140개 이상의 언어를 지원합니다. 🚀",
      tokens: [
        { t: "안녕하세요", id: 61659, cat: "word", verified: true, note: "Entire 5-syllable Korean greeting is 1 token in Gemma 4's 262K vocab!" },
        { t: "!", id: 236888, cat: "code", verified: true, note: "Exclamation mark" },
        { t: "▁Gemma", id: 147224, cat: "word", verified: true, note: "Space + Gemma" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Space before digit" },
        { t: "4", id: 236812, cat: "digit", verified: true, note: "Digit '4'" },
        { t: "는", id: 237170, cat: "subword", verified: true, note: "Korean topic particle attached to 4" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Space before number 140" },
        { t: "1", id: 236770, cat: "digit", verified: true, note: "Digit '1'" },
        { t: "4", id: 236812, cat: "digit", verified: true, note: "Digit '4'" },
        { t: "0", id: 236771, cat: "digit", verified: true, note: "Digit '0'" },
        { t: "개", id: 238242, cat: "subword", verified: true, note: "Korean counter morpheme" },
        { t: "▁이상의", id: 183126, cat: "word", verified: true, note: "Korean phrase 'more than'" },
        { t: "▁언", id: 64822, cat: "subword", verified: true, note: "First subword of 'language' (언어)" },
        { t: "어를", id: 120857, cat: "subword", verified: true, note: "Second subword + object marker" },
        { t: "▁지원", id: 65363, cat: "word", verified: true, note: "Root noun 'support'" },
        { t: "합니다", id: 19773, cat: "subword", verified: true, note: "Polite verb ending" },
        { t: ".", id: 236761, cat: "code", verified: true, note: "Period" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Space before emoji" },
        { t: "🚀", id: 242015, cat: "subword", verified: true, note: "Single dedicated emoji token (no UTF-8 byte fallback needed)" },
      ],
    },
    {
      label: "Rare Words & Byte Fallback",
      text: "똠얌꿍 / 加鬮さん",
      tokens: [
        { t: "<0xEB>", raw: "똠[0xEB]", id: 473, cat: "subword", verified: true, note: "UTF-8 byte 1/3 of rare syllable '똠' (U+B620 → 0xEB 0x98 0xA0)" },
        { t: "<0x98>", raw: "똠[0x98]", id: 390, cat: "subword", verified: true, note: "UTF-8 byte 2/3 of rare syllable '똠' (0x98)" },
        { t: "<0xA0>", raw: "똠[0xA0]", id: 398, cat: "subword", verified: true, note: "UTF-8 byte 3/3 of rare syllable '똠' (0xA0) — never collapses to <unk>!" },
        { t: "얌", id: 249170, cat: "subword", verified: true, note: "Known Korean syllable '얌' in Gemma 4's 262K vocabulary" },
        { t: "꿍", id: 249927, cat: "subword", verified: true, note: "Known Korean syllable '꿍' in Gemma 4's vocabulary" },
        { t: "▁/", id: 965, cat: "code", verified: true, note: "Space + slash separator token" },
        { t: "▁加", id: 50624, cat: "word", verified: true, note: "Leading space + known kanji '加'" },
        { t: "<0xE9>", raw: "鬮[0xE9]", id: 471, cat: "subword", verified: true, note: "UTF-8 byte 1/3 of ultra-rare kanji '鬮' (U+9B2E → 0xE9 0xAC 0xAE)" },
        { t: "<0xAC>", raw: "鬮[0xAC]", id: 410, cat: "subword", verified: true, note: "UTF-8 byte 2/3 of rare kanji '鬮' (0xAC)" },
        { t: "<0xAE>", raw: "鬮[0xAE]", id: 412, cat: "subword", verified: true, note: "UTF-8 byte 3/3 of rare kanji '鬮' (0xAE)" },
        { t: "さん", id: 9821, cat: "subword", verified: true, note: "Common 2-hiragana honorific suffix 'さん' merged as 1 token" },
      ],
    },
    {
      label: "Reserved <unusedXX> Tokens",
      text: "<unused0>citation<unused1> Gemma 4",
      tokens: [
        { t: "<unused0>", id: 6, cat: "control", verified: true, note: "Reserved <unused0> slot (1 of 6,227) — repurpose for custom fine-tuning tags without resizing the 262,144 × d_model matrix!" },
        { t: "citation", id: 99697, cat: "word", verified: true, note: "Custom domain tag or structured payload inside reserved control markers" },
        { t: "<unused1>", id: 7, cat: "control", verified: true, note: "Reserved <unused1> slot — pre-allocated in the vocabulary & embedding table" },
        { t: "▁Gemma", id: 147224, cat: "word", verified: true, note: "Space + model name" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Space before version digit" },
        { t: "4", id: 236812, cat: "digit", verified: true, note: "Digit '4'" },
      ],
    },
    {
      label: "Python Code",
      text: "def fibonacci(n):\n    return n if n <= 1",
      tokens: [
        { t: "def", id: 2063, cat: "code", verified: true, note: "Python keyword token" },
        { t: "▁fib", id: 10779, cat: "subword", verified: true, note: "Space + common identifier prefix" },
        { t: "onacci", id: 78113, cat: "subword", verified: true, note: "Subword suffix completing 'fibonacci'" },
        { t: "(", id: 236769, cat: "code", verified: true, note: "Open parenthesis" },
        { t: "n", id: 236749, cat: "code", verified: true, note: "Parameter identifier" },
        { t: "):", id: 1473, cat: "code", verified: true, note: "Merged closing paren + colon token for Python!" },
        { t: "\\n", id: 107, cat: "control", verified: true, note: "Newline character token" },
        { t: "▁▁▁▁", id: 140, cat: "code", verified: true, note: "4-space Python indentation compressed into a SINGLE token!" },
        { t: "return", id: 2060, cat: "code", verified: true, note: "Python return keyword" },
        { t: "▁n", id: 538, cat: "code", verified: true, note: "Space + variable n" },
        { t: "▁if", id: 768, cat: "code", verified: true, note: "Space + conditional keyword" },
        { t: "▁n", id: 538, cat: "code", verified: true, note: "Space + variable n" },
        { t: "▁<=", id: 6605, cat: "code", verified: true, note: "Space + comparison operator" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Space before numeric literal" },
        { t: "1", id: 236770, cat: "digit", verified: true, note: "Digit '1'" },
      ],
    },
    {
      label: "Chat Control Tokens",
      text: "<|turn>user\n<|think|>Hello Gemma 4!<turn|>",
      tokens: [
        { t: "<|turn>", id: 105, cat: "control", verified: true, note: "Special control token starting a turn" },
        { t: "user", id: 2364, cat: "word", verified: true, note: "Role identifier token" },
        { t: "\\n", id: 107, cat: "control", verified: true, note: "Newline separator after role" },
        { t: "<|think|>", id: 98, cat: "control", verified: true, note: "Special Thinking Mode trigger token" },
        { t: "Hello", id: 9259, cat: "word", verified: true, note: "Greeting word token" },
        { t: "▁Gemma", id: 147224, cat: "word", verified: true, note: "Space + model name" },
        { t: "▁", id: 236743, cat: "subword", verified: true, note: "Space before version digit" },
        { t: "4", id: 236812, cat: "digit", verified: true, note: "Digit '4'" },
        { t: "!", id: 236888, cat: "code", verified: true, note: "Exclamation mark" },
        { t: "<turn|>", id: 106, cat: "control", verified: true, note: "Turn end / stop control token" },
      ],
    },
    {
      label: "Compound Subwords",
      text: "Antidisestablishmentarianism and supercalifragilisticexpialidocious",
      tokens: [
        { t: "Ant", id: 14054, cat: "subword", verified: true, note: "Prefix piece 1" },
        { t: "idis", id: 33090, cat: "subword", verified: true, note: "Subword piece 2" },
        { t: "establishment", id: 163524, cat: "subword", verified: true, note: "Root word piece inside compound word!" },
        { t: "arian", id: 8225, cat: "subword", verified: true, note: "Suffix piece 1" },
        { t: "ism", id: 1929, cat: "subword", verified: true, note: "Noun suffix piece 2" },
        { t: "▁and", id: 532, cat: "word", verified: true, note: "Common conjunction" },
        { t: "▁super", id: 2393, cat: "subword", verified: true, note: "Space + common prefix 'super'" },
        { t: "cal", id: 1653, cat: "subword", verified: true, note: "Syllable subword" },
        { t: "if", id: 584, cat: "subword", verified: true, note: "Syllable subword" },
        { t: "rag", id: 16100, cat: "subword", verified: true, note: "Syllable subword" },
        { t: "il", id: 543, cat: "subword", verified: true, note: "Syllable subword" },
        { t: "istic", id: 3316, cat: "subword", verified: true, note: "Adjective suffix piece" },
        { t: "exp", id: 4212, cat: "subword", verified: true, note: "Syllable subword" },
        { t: "ial", id: 706, cat: "subword", verified: true, note: "Syllable subword" },
        { t: "id", id: 547, cat: "subword", verified: true, note: "Syllable subword" },
        { t: "ocious", id: 100466, cat: "subword", verified: true, note: "Ending subword — zero <unk> errors on any rare word!" },
      ],
    },
  ];

  const hashTokenId = (str) => {
    if (KNOWN_VOCAB[str] !== undefined) return KNOWN_VOCAB[str];
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return 1200 + (Math.abs(h) % 245000);
  };

  const decomposeKnownSubwords = (display) => {
    let best2 = null;
    let best2Score = Infinity;
    for (let i = 1; i < display.length; i++) {
      const p1 = display.slice(0, i);
      const p2 = display.slice(i);
      if (p2.length < 2) continue;
      if (p1.length < 2 && !/^[A-Z]$/.test(p1)) continue;
      const id1 = KNOWN_VOCAB[p1];
      const id2 = KNOWN_VOCAB[p2];
      if (id1 !== undefined && id2 !== undefined && id1 + id2 < best2Score) {
        best2Score = id1 + id2;
        best2 = [
          { t: p1, id: id1, cat: "subword", verified: true, note: "Verified Gemma 4 BPE subword prefix/root" },
          { t: p2, id: id2, cat: "subword", verified: true, note: "Verified Gemma 4 BPE subword continuation" },
        ];
      }
    }
    if (best2) return best2;

    let best3 = null;
    let best3Score = Infinity;
    for (let i = 2; i < display.length - 3; i++) {
      const p1 = display.slice(0, i);
      const id1 = KNOWN_VOCAB[p1];
      if (id1 === undefined) continue;
      for (let j = i + 2; j < display.length - 1; j++) {
        const p2 = display.slice(i, j);
        const p3 = display.slice(j);
        const id2 = KNOWN_VOCAB[p2];
        const id3 = KNOWN_VOCAB[p3];
        if (id2 !== undefined && id3 !== undefined && id1 + id2 + id3 < best3Score) {
          best3Score = id1 + id2 + id3;
          best3 = [
            { t: p1, id: id1, cat: "subword", verified: true, note: "Verified Gemma 4 BPE subword piece 1/3" },
            { t: p2, id: id2, cat: "subword", verified: true, note: "Verified Gemma 4 BPE subword piece 2/3" },
            { t: p3, id: id3, cat: "subword", verified: true, note: "Verified Gemma 4 BPE subword piece 3/3" },
          ];
        }
      }
    }
    return best3;
  };

  const COMMON_SUFFIXES = [
    "ization", "ation", "ition", "tion", "sion", "ment", "ness", "able", "ible",
    "ious", "eous", "ical", "istic", "ing", "ers", "est", "ful", "less", "ive",
    "ous", "ize", "ism", "ist", "ity", "ly", "ed", "er", "al", "ic", "es", "s",
  ];

  const tokenizeOfflineFallback = (text) => {
    const preset = PRESET_TOKENIZATIONS.find((p) => p.text === text);
    if (preset) return preset.tokens;

    const out = [];
    const pattern = /(<unused\d+>|<0x[0-9A-Fa-f]{2}>|<\|[a-z_"]+\|>|<\|[a-z_]+>|<[a-z_]+\|>|<pad>|<eos>|<bos>|<unk>|<mask>|\[multimodal\]|    |\n|\d| ?안녕하세요| ?이상의| ?어를| ?지원| ?합니다| ?さん| ?[A-Za-z]+| ?[^\sA-Za-z\d])/g;
    const matches = text.match(pattern) || [];

    for (const raw of matches) {
      const unusedMatch = /^<unused(\d+)>$/.exec(raw);
      const byteMatch = /^<0x([0-9A-Fa-f]{2})>$/.exec(raw);
      if (unusedMatch) {
        const uNum = Number(unusedMatch[1]);
        const uId = getUnusedTokenId(uNum);
        if (uId !== null) {
          out.push({
            t: raw,
            id: uId,
            cat: "control",
            verified: true,
            note: `Reserved ${raw} token (1 of 6,227 unused slots) for custom fine-tuning without resizing the embedding matrix`,
          });
          continue;
        }
      }
      if (byteMatch) {
        const byteVal = parseInt(byteMatch[1].toUpperCase(), 16);
        const bTok = `<0x${byteMatch[1].toUpperCase()}>`;
        out.push({
          t: bTok,
          id: 238 + byteVal,
          cat: "subword",
          verified: true,
          note: `UTF-8 raw byte fallback token ${bTok}`,
        });
        continue;
      }
      if (CONTROL_TOKENS[raw]) {
        out.push({ t: raw, id: CONTROL_TOKENS[raw].id, cat: "control", verified: true, note: CONTROL_TOKENS[raw].note });
      } else if (raw === "\n") {
        out.push({ t: "\\n", id: 107, cat: "control", verified: true, note: "Newline token" });
      } else if (raw === "    ") {
        out.push({ t: "▁▁▁▁", id: 140, cat: "code", verified: true, note: "4-space indentation token" });
      } else if (DIGIT_IDS[raw] !== undefined) {
        out.push({ t: raw, id: DIGIT_IDS[raw], cat: "digit", verified: true, note: `Single digit '${raw}' token` });
      } else {
        const hasSpace = raw.startsWith(" ");
        const clean = hasSpace ? raw.slice(1) : raw;
        if (!clean) {
          out.push({ t: "▁", id: 236743, cat: "subword", verified: true, note: "Standalone whitespace token" });
          continue;
        }
        if (BYTE_FALLBACK_CHARS[clean]) {
          if (hasSpace) {
            out.push({ t: "▁", id: 236743, cat: "subword", verified: true, note: "Leading space before byte-fallback character" });
          }
          for (const bTok of BYTE_FALLBACK_CHARS[clean]) {
            out.push(bTok);
          }
          continue;
        }
        const display = (hasSpace ? "▁" : "") + clean;
        const isPunct = /^[^\w\u00C0-\uFFFF]+$/.test(clean);
        if (KNOWN_VOCAB[display] !== undefined) {
          out.push({
            t: display,
            id: KNOWN_VOCAB[display],
            cat: isPunct ? "code" : "word",
            verified: true,
            note: isPunct
              ? "Verified Gemma 4 punctuation / operator token"
              : "Verified Gemma 4 vocabulary token",
          });
          continue;
        }
        if (hasSpace && KNOWN_VOCAB[clean] !== undefined) {
          out.push({ t: "▁", id: 236743, cat: "subword", verified: true, note: "Leading space token" });
          out.push({
            t: clean,
            id: KNOWN_VOCAB[clean],
            cat: isPunct ? "code" : "word",
            verified: true,
            note: "Verified Gemma 4 vocabulary token",
          });
          continue;
        }
        const decomposed = decomposeKnownSubwords(display);
        if (decomposed) {
          for (const piece of decomposed) out.push(piece);
          continue;
        }
        let matchedSuffix = null;
        if (/^[A-Za-z]+$/.test(clean) && clean.length >= 6) {
          for (const sfx of COMMON_SUFFIXES) {
            if (clean.endsWith(sfx) && clean.length - sfx.length >= 3 && KNOWN_VOCAB[sfx] !== undefined) {
              matchedSuffix = sfx;
              break;
            }
          }
        }
        if (matchedSuffix) {
          const stem = (hasSpace ? "▁" : "") + clean.slice(0, clean.length - matchedSuffix.length);
          const stemKnown = KNOWN_VOCAB[stem] !== undefined;
          out.push({
            t: stem,
            id: hashTokenId(stem),
            cat: "subword",
            verified: stemKnown,
            note: stemKnown
              ? "Verified Gemma 4 subword root"
              : "Offline fallback estimate (custom word root outside embedded subset)",
          });
          out.push({
            t: matchedSuffix,
            id: KNOWN_VOCAB[matchedSuffix],
            cat: "subword",
            verified: true,
            note: "Verified Gemma 4 subword suffix",
          });
        } else if (clean.length <= 7 || !/^[A-Za-z]+$/.test(clean)) {
          out.push({
            t: display,
            id: hashTokenId(display),
            cat: isPunct ? "code" : "word",
            verified: false,
            note: "Offline fallback estimate (custom token outside embedded vocabulary subset)",
          });
        } else {
          const mid = Math.ceil(clean.length * 0.55);
          const p1 = (hasSpace ? "▁" : "") + clean.slice(0, mid);
          const p2 = clean.slice(mid);
          const p1Known = KNOWN_VOCAB[p1] !== undefined;
          const p2Known = KNOWN_VOCAB[p2] !== undefined;
          out.push({
            t: p1,
            id: hashTokenId(p1),
            cat: "subword",
            verified: p1Known,
            note: p1Known ? "Verified Gemma 4 subword piece" : "Offline fallback estimate (leading subword piece)",
          });
          out.push({
            t: p2,
            id: hashTokenId(p2),
            cat: "subword",
            verified: p2Known,
            note: p2Known ? "Verified Gemma 4 subword piece" : "Offline fallback estimate (trailing subword piece)",
          });
        }
      }
    }
    return out;
  };

  /* --- Real Gemma 4 Tokenizer Client (@huggingface/tokenizers + google/gemma-4-E2B-it) --- */
  const GEMMA4_TOKENIZER_URLS = {
    jsCdns: [
      "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs",
      "https://unpkg.com/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs",
    ],
    tokenizerJson: "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/tokenizer.json",
    tokenizerConfig: "https://huggingface.co/google/gemma-4-E2B-it/resolve/main/tokenizer_config.json",
    expectedBytes: 32169626,
    idbName: "gemma4-llm-demo-cache",
    idbStore: "tokenizer-artifacts",
    idbKey: "google/gemma-4-E2B-it@v1",
  };

  const CODE_KEYWORDS = new Set([
    "def", "return", "import", "elif", "else", "while", "lambda", "yield",
    "async", "await", "const", "let", "var", "function", "True", "False", "None",
  ]);

  const utf8Decoder = typeof TextDecoder !== "undefined" ? new TextDecoder("utf-8", { fatal: false }) : null;

  const utf8SeqLen = (leadByte) => {
    if ((leadByte & 0x80) === 0x00) return 1;
    if ((leadByte & 0xE0) === 0xC0) return 2;
    if ((leadByte & 0xF0) === 0xE0) return 3;
    if ((leadByte & 0xF8) === 0xF0) return 4;
    return 1;
  };

  const annotateByteFallbacks = (rawTokens, ids) => {
    const byteAnnotations = new Map();
    let i = 0;
    while (i < rawTokens.length) {
      if (!/^<0x([0-9A-Fa-f]{2})>$/.test(rawTokens[i])) {
        i++;
        continue;
      }
      let end = i;
      while (end < rawTokens.length && /^<0x([0-9A-Fa-f]{2})>$/.test(rawTokens[end])) {
        end++;
      }
      let cursor = i;
      while (cursor < end) {
        const leadVal = parseInt(rawTokens[cursor].slice(3, 5), 16);
        const expected = Math.min(end - cursor, utf8SeqLen(leadVal));
        let groupLen = 1;
        while (
          groupLen < expected &&
          (parseInt(rawTokens[cursor + groupLen].slice(3, 5), 16) & 0xC0) === 0x80
        ) {
          groupLen++;
        }
        const bytes = new Uint8Array(groupLen);
        const hexList = [];
        for (let k = 0; k < groupLen; k++) {
          const hex = rawTokens[cursor + k].slice(3, 5).toUpperCase();
          bytes[k] = parseInt(hex, 16);
          hexList.push(`0x${hex}`);
        }
        const ch = utf8Decoder ? utf8Decoder.decode(bytes) : "";
        const validChar = ch && !ch.includes("\uFFFD") ? ch : null;
        const cp = validChar ? validChar.codePointAt(0) : null;
        const cpHex = cp !== null ? `U+${cp.toString(16).toUpperCase().padStart(4, "0")}` : "raw UTF-8";
        for (let k = 0; k < groupLen; k++) {
          const idx = cursor + k;
          const hex = rawTokens[idx].slice(3, 5).toUpperCase();
          const id = ids[idx];
          byteAnnotations.set(idx, {
            t: `<0x${hex}>`,
            raw: validChar ? `${validChar}[0x${hex}]` : `<0x${hex}>`,
            id,
            cat: "subword",
            verified: true,
            note: validChar
              ? `UTF-8 byte ${k + 1}/${groupLen} of '${validChar}' (${cpHex} → ${hexList.join(" ")}) — zero <unk> loss!`
              : `UTF-8 raw byte fallback token <0x${hex}>`,
          });
        }
        cursor += groupLen;
      }
      i = end;
    }
    return byteAnnotations;
  };

  const isWordLikePiece = (p) => {
    if (!p || CONTROL_TOKENS[p] || /^<unused\d+>$/.test(p) || /^[\n\t]+$/.test(p) || /^▁+$/.test(p)) {
      return false;
    }
    if (/^<0x[0-9A-Fa-f]{2}>$/.test(p)) return true;
    const c = p.replace(/^▁/, "");
    return /[\p{L}\p{N}]/u.test(c);
  };

  const buildRichTokens = (text, rawTokens, ids) => {
    const preset = PRESET_TOKENIZATIONS.find((p) => p.text === text);
    if (
      preset &&
      preset.tokens.length === ids.length &&
      preset.tokens.every((pt, idx) => pt.id === ids[idx])
    ) {
      return preset.tokens.map((pt, idx) => ({ ...pt, id: ids[idx], verified: true }));
    }

    const byteMap = annotateByteFallbacks(rawTokens, ids);
    return rawTokens.map((piece, i) => {
      const id = ids[i];
      if (byteMap.has(i)) return byteMap.get(i);
      if (CONTROL_TOKENS[piece]) {
        return { t: piece, raw: piece, id, cat: "control", verified: true, note: CONTROL_TOKENS[piece].note };
      }
      if (/^<unused(\d+)>$/.test(piece)) {
        return {
          t: piece,
          raw: piece,
          id,
          cat: "control",
          verified: true,
          note: `Reserved ${piece} slot (1 of 6,227) — repurpose for custom fine-tuning tags without resizing the 262,144 × d_model matrix!`,
        };
      }
      if (/^\n+$/.test(piece)) {
        const n = piece.length;
        return {
          t: n <= 3 ? "\\n".repeat(n) : `\\n×${n}`,
          raw: "↵\n".repeat(n),
          id,
          cat: "control",
          verified: true,
          note: n === 1 ? "Newline character token" : `${n} consecutive newlines merged into a single token`,
        };
      }
      if (/^[0-9]$/.test(piece)) {
        return {
          t: piece,
          raw: piece,
          id,
          cat: "digit",
          verified: true,
          note: `Individual digit '${piece}' token (Gemma 4 splits all digits 0–9 for consistent arithmetic)`,
        };
      }
      if (piece === "▁") {
        return {
          t: "▁",
          raw: "␣",
          id,
          cat: "subword",
          verified: true,
          note: "Standalone space token ▁ (U+2581)",
        };
      }
      if (/^▁{2,}$/.test(piece)) {
        const n = piece.length;
        return {
          t: n <= 8 ? piece : `▁×${n}`,
          raw: "·".repeat(n),
          id,
          cat: "code",
          verified: true,
          note: `${n}-space indentation compressed into a SINGLE token!`,
        };
      }
      if (/^\t+$/.test(piece)) {
        const n = piece.length;
        return {
          t: n <= 3 ? "\\t".repeat(n) : `\\t×${n}`,
          raw: "⇥".repeat(n),
          id,
          cat: "code",
          verified: true,
          note: n === 1 ? "Tab character token" : `${n}-tab indentation compressed into a single token`,
        };
      }
      const clean = piece.replace(/^▁/, "");
      if (/^\p{Extended_Pictographic}/u.test(clean)) {
        return {
          t: piece,
          raw: piece.replace(/▁/g, "␣"),
          id,
          cat: "subword",
          verified: true,
          note: "Dedicated emoji token in Gemma 4's 262K vocabulary (no UTF-8 byte fallback needed)",
        };
      }
      if (clean.length > 0 && !/[\p{L}\p{N}]/u.test(clean)) {
        return {
          t: piece,
          raw: piece.replace(/▁/g, "␣"),
          id,
          cat: "code",
          verified: true,
          note: piece.startsWith("▁")
            ? "Space + punctuation / operator token"
            : "Punctuation / operator token",
        };
      }
      const prevP = i > 0 ? rawTokens[i - 1] : null;
      const nextP = i + 1 < rawTokens.length ? rawTokens[i + 1] : null;
      const hasLeftAttach = !piece.startsWith("▁") && Boolean(prevP && isWordLikePiece(prevP));
      const hasRightAttach = Boolean(nextP && !nextP.startsWith("▁") && isWordLikePiece(nextP));
      if (hasLeftAttach || hasRightAttach) {
        return {
          t: piece,
          raw: piece.replace(/▁/g, "␣"),
          id,
          cat: "subword",
          verified: true,
          note: !hasLeftAttach
            ? "Initial subword piece from Gemma 4's 262K vocabulary"
            : "Subword continuation piece from Gemma 4's 262K vocabulary",
        };
      }
      if (CODE_KEYWORDS.has(clean)) {
        return {
          t: piece,
          raw: piece.replace(/▁/g, "␣"),
          id,
          cat: "code",
          verified: true,
          note: "Programming keyword token in Gemma 4's vocabulary",
        };
      }
      return {
        t: piece,
        raw: piece.replace(/▁/g, "␣"),
        id,
        cat: "word",
        verified: true,
        note: piece.startsWith("▁")
          ? "Whole word with leading space marker ▁ (U+2581)"
          : "Whole word token in Gemma 4's 262K vocabulary",
      };
    });
  };

  const createRealTokenizerService = () => {
    let status = "idle"; // "idle" | "loading" | "ready" | "error"
    let statusLabel = "Loading Gemma 4 tokenizer…";
    let worker = null;
    let mainTokenizer = null;
    let initPromise = null;
    let reqCounter = 0;
    const pendingReqs = new Map();
    const subscribers = new Set();

    const notify = () => {
      for (const fn of subscribers) fn({ status, statusLabel });
    };

    const setStatus = (nextStatus, nextLabel) => {
      status = nextStatus;
      statusLabel = nextLabel;
      notify();
    };

    const idbRead = async () => {
      try {
        if (typeof indexedDB === "undefined") return null;
        return await new Promise((resolve) => {
          const openReq = indexedDB.open(GEMMA4_TOKENIZER_URLS.idbName, 1);
          openReq.onupgradeneeded = () => {
            const db = openReq.result;
            if (!db.objectStoreNames.contains(GEMMA4_TOKENIZER_URLS.idbStore)) {
              db.createObjectStore(GEMMA4_TOKENIZER_URLS.idbStore);
            }
          };
          openReq.onerror = () => resolve(null);
          openReq.onsuccess = () => {
            try {
              const db = openReq.result;
              const tx = db.transaction(GEMMA4_TOKENIZER_URLS.idbStore, "readonly");
              const getReq = tx.objectStore(GEMMA4_TOKENIZER_URLS.idbStore).get(GEMMA4_TOKENIZER_URLS.idbKey);
              getReq.onsuccess = () => {
                db.close();
                resolve(getReq.result || null);
              };
              getReq.onerror = () => {
                db.close();
                resolve(null);
              };
            } catch (_) {
              resolve(null);
            }
          };
        });
      } catch (_) {
        return null;
      }
    };

    const idbWrite = async (payload) => {
      try {
        if (typeof indexedDB === "undefined") return;
        await new Promise((resolve) => {
          const openReq = indexedDB.open(GEMMA4_TOKENIZER_URLS.idbName, 1);
          openReq.onupgradeneeded = () => {
            const db = openReq.result;
            if (!db.objectStoreNames.contains(GEMMA4_TOKENIZER_URLS.idbStore)) {
              db.createObjectStore(GEMMA4_TOKENIZER_URLS.idbStore);
            }
          };
          openReq.onerror = () => resolve();
          openReq.onsuccess = () => {
            try {
              const db = openReq.result;
              const tx = db.transaction(GEMMA4_TOKENIZER_URLS.idbStore, "readwrite");
              tx.objectStore(GEMMA4_TOKENIZER_URLS.idbStore).put(payload, GEMMA4_TOKENIZER_URLS.idbKey);
              tx.oncomplete = () => {
                db.close();
                resolve();
              };
              tx.onerror = () => {
                db.close();
                resolve();
              };
            } catch (_) {
              resolve();
            }
          };
        });
      } catch (_) {
        /* ignore quota / private-mode errors */
      }
    };

    const fetchTokenizerArtifacts = async (onProgress) => {
      const cached = await idbRead();
      if (cached && cached.jsonBytes && cached.configText) {
        onProgress("Initializing cached Gemma 4 BPE…");
        return cached;
      }
      onProgress("Downloading tokenizer.json (0%)…");
      const [tokRes, cfgRes] = await Promise.all([
        fetch(GEMMA4_TOKENIZER_URLS.tokenizerJson),
        fetch(GEMMA4_TOKENIZER_URLS.tokenizerConfig),
      ]);
      if (!tokRes.ok || !cfgRes.ok) {
        throw new Error(`HTTP ${tokRes.status}/${cfgRes.status}`);
      }
      const configText = await cfgRes.text();
      const total = Number(tokRes.headers.get("content-length")) || GEMMA4_TOKENIZER_URLS.expectedBytes;
      let jsonBytes;
      if (tokRes.body && typeof tokRes.body.getReader === "function") {
        const reader = tokRes.body.getReader();
        const chunks = [];
        let loaded = 0;
        let lastTick = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          loaded += value.byteLength;
          const now = Date.now();
          if (now - lastTick > 80) {
            lastTick = now;
            const pct = Math.min(99, Math.round((loaded / total) * 100));
            onProgress(`Downloading tokenizer.json (${pct}%)…`);
          }
        }
        jsonBytes = new Uint8Array(loaded);
        let offset = 0;
        for (const c of chunks) {
          jsonBytes.set(c, offset);
          offset += c.byteLength;
        }
      } else {
        jsonBytes = new Uint8Array(await tokRes.arrayBuffer());
      }
      const payload = { jsonBytes, configText };
      idbWrite(payload);
      return payload;
    };

    const importTokenizerClass = async () => {
      let lastErr = null;
      for (const url of GEMMA4_TOKENIZER_URLS.jsCdns) {
        try {
          const mod = await import(url);
          if (mod && mod.Tokenizer) return mod.Tokenizer;
        } catch (err) {
          lastErr = err;
        }
      }
      throw lastErr || new Error("Unable to load @huggingface/tokenizers");
    };

    const withSentencePieceSpecialSlots = (tj) => {
      const existingIds = new Set((tj.added_tokens || []).map((t) => t.id));
      const extraAdded = [];
      const vocab = (tj.model && tj.model.vocab) || {};
      for (const [piece, id] of Object.entries(vocab)) {
        if (!existingIds.has(id) && (/^<unused\d+>$/.test(piece) || piece === "[multimodal]")) {
          extraAdded.push({
            id,
            content: piece,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: true,
          });
        }
      }
      return { ...tj, added_tokens: [...(tj.added_tokens || []), ...extraAdded] };
    };

    const startWorkerEngine = (artifacts) =>
      new Promise((resolve, reject) => {
        if (typeof Worker === "undefined" || typeof Blob === "undefined" || typeof URL === "undefined") {
          reject(new Error("Web Worker unavailable"));
          return;
        }
        const workerCode = `
          const CDNS = ${JSON.stringify(GEMMA4_TOKENIZER_URLS.jsCdns)};
          let tokenizer = null;
          const loadLib = async () => {
            let errLast = null;
            for (const u of CDNS) {
              try {
                const m = await import(u);
                if (m && m.Tokenizer) return m.Tokenizer;
              } catch (e) { errLast = e; }
            }
            throw errLast || new Error("CDN import failed");
          };
          self.onmessage = async (ev) => {
            const msg = ev.data;
            if (msg.type === "init") {
              try {
                const Tokenizer = await loadLib();
                const tj = JSON.parse(new TextDecoder("utf-8").decode(msg.jsonBytes));
                const tc = JSON.parse(msg.configText);
                const existingIds = new Set((tj.added_tokens || []).map((t) => t.id));
                const extra = [];
                for (const [piece, id] of Object.entries((tj.model && tj.model.vocab) || {})) {
                  if (!existingIds.has(id) && (/^<unused\\d+>$/.test(piece) || piece === "[multimodal]")) {
                    extra.push({ id, content: piece, single_word: false, lstrip: false, rstrip: false, normalized: false, special: true });
                  }
                }
                tokenizer = new Tokenizer({ ...tj, added_tokens: [...(tj.added_tokens || []), ...extra] }, tc);
                self.postMessage({ type: "ready" });
              } catch (e) {
                self.postMessage({ type: "init-error", message: String(e && e.message ? e.message : e) });
              }
              return;
            }
            if (msg.type === "encode") {
              if (!tokenizer || !msg.text) {
                self.postMessage({ type: "encoded", reqId: msg.reqId, tokens: [], ids: [] });
                return;
              }
              const enc = tokenizer.encode(msg.text, { add_special_tokens: false });
              self.postMessage({ type: "encoded", reqId: msg.reqId, tokens: enc.tokens, ids: enc.ids });
            }
          };
        `;
        let blobUrl = null;
        try {
          blobUrl = URL.createObjectURL(new Blob([workerCode], { type: "text/javascript" }));
          const w = new Worker(blobUrl, { type: "module" });
          let settled = false;
          w.onerror = (err) => {
            if (blobUrl) {
              URL.revokeObjectURL(blobUrl);
              blobUrl = null;
            }
            if (!settled) {
              settled = true;
              w.terminate();
              reject(err || new Error("Worker failed"));
            }
          };
          w.onmessage = (ev) => {
            const msg = ev.data || {};
            if (msg.type === "ready") {
              if (blobUrl) {
                URL.revokeObjectURL(blobUrl);
                blobUrl = null;
              }
              settled = true;
              worker = w;
              resolve();
              return;
            }
            if (msg.type === "init-error") {
              if (blobUrl) {
                URL.revokeObjectURL(blobUrl);
                blobUrl = null;
              }
              if (!settled) {
                settled = true;
                w.terminate();
                reject(new Error(msg.message || "Worker init error"));
              }
              return;
            }
            if (msg.type === "encoded") {
              const resolver = pendingReqs.get(msg.reqId);
              if (resolver) {
                pendingReqs.delete(msg.reqId);
                resolver({ tokens: msg.tokens, ids: msg.ids });
              }
            }
          };
          const copyBuf = artifacts.jsonBytes.slice().buffer;
          w.postMessage({ type: "init", jsonBytes: copyBuf, configText: artifacts.configText }, [copyBuf]);
        } catch (err) {
          if (blobUrl) URL.revokeObjectURL(blobUrl);
          reject(err);
        }
      });

    const startMainThreadEngine = async (artifacts) => {
      const Tokenizer = await importTokenizerClass();
      await new Promise((r) => setTimeout(r, 0));
      const tj = JSON.parse(new TextDecoder("utf-8").decode(artifacts.jsonBytes));
      const tc = JSON.parse(artifacts.configText);
      await new Promise((r) => setTimeout(r, 0));
      mainTokenizer = new Tokenizer(withSentencePieceSpecialSlots(tj), tc);
    };

    const ensureReady = () => {
      if (status === "ready") return Promise.resolve(true);
      if (initPromise) return initPromise;
      setStatus("loading", "Loading Gemma 4 tokenizer…");
      initPromise = (async () => {
        try {
          const artifacts = await fetchTokenizerArtifacts((msg) => setStatus("loading", msg));
          setStatus("loading", "Building 262K BPE table…");
          try {
            await startWorkerEngine(artifacts);
          } catch (_) {
            await startMainThreadEngine(artifacts);
          }
          setStatus("ready", "● Live: google/gemma-4-E2B-it");
          return true;
        } catch (_) {
          initPromise = null;
          setStatus("error", "⚠ Offline subset · Click to retry");
          return false;
        }
      })();
      return initPromise;
    };

    const encode = async (text) => {
      if (!text) return [];
      if (status === "ready") {
        if (worker) {
          const reqId = ++reqCounter;
          const res = await new Promise((resolve) => {
            pendingReqs.set(reqId, resolve);
            worker.postMessage({ type: "encode", reqId, text });
          });
          return buildRichTokens(text, res.tokens, res.ids);
        }
        if (mainTokenizer) {
          const enc = mainTokenizer.encode(text, { add_special_tokens: false });
          return buildRichTokens(text, enc.tokens, enc.ids);
        }
      }
      return tokenizeOfflineFallback(text);
    };

    return {
      getState: () => ({ status, statusLabel }),
      subscribe: (fn) => {
        subscribers.add(fn);
        return () => subscribers.delete(fn);
      },
      ensureReady,
      retry: () => {
        if (status === "loading") return ensureReady();
        initPromise = null;
        return ensureReady();
      },
      encode,
    };
  };

  const realTokenizerService = createRealTokenizerService();

  const TOKEN_CAT_META = {
    word: {
      name: "Whole Word Token",
      bg: "var(--gemma-cat-turn-bg)",
      fg: "var(--gemma-cat-turn-fg)",
      border: "var(--gemma-cat-turn-border)",
    },
    subword: {
      name: "Subword Piece",
      bg: "var(--gemma-cat-multi-bg)",
      fg: "var(--gemma-cat-multi-fg)",
      border: "var(--gemma-cat-multi-border)",
    },
    digit: {
      name: "Individual Digit Token",
      bg: "var(--gemma-cat-think-bg)",
      fg: "var(--gemma-cat-think-fg)",
      border: "var(--gemma-cat-think-border)",
    },
    control: {
      name: "Special Control Token",
      bg: "var(--gemma-cat-role-bg)",
      fg: "var(--gemma-cat-role-fg)",
      border: "var(--gemma-cat-role-border)",
    },
    code: {
      name: "Code / Punctuation Token",
      bg: "var(--gemma-cat-tool-bg)",
      fg: "var(--gemma-cat-tool-fg)",
      border: "var(--gemma-cat-tool-border)",
    },
  };

  const mountTokenizer = (el) => {
    let currentText = PRESET_TOKENIZATIONS[0].text;
    let currentTokens = PRESET_TOKENIZATIONS[0].tokens;
    let selectedIdx = 0;
    let pinnedIdx = null;
    let renderedTokensRef = null;
    let animateEntrance = true;
    let showScanner = false;
    let encodeSeq = 0;

    el.innerHTML = `
      <div class="llm-widget__header">
        <p class="llm-widget__title">Interactive: Gemma 4 SentencePiece Tokenizer</p>
        <div class="llm-pill-row" style="margin-bottom:0">
          <span class="llm-widget__badge llm-widget__badge--loading" data-role="engine-status" title="Real google/gemma-4-E2B-it tokenizer via @huggingface/tokenizers">Loading Gemma 4 tokenizer…</span>
          <span class="llm-widget__badge">Vocab: 262,144 tokens</span>
        </div>
      </div>
      <div class="llm-toolbar llm-toolbar--compact" role="group" aria-label="Select text scenario">
        <span class="llm-toolbar__label">Scenario</span>
        <div class="llm-toolbar__group" data-role="presets"></div>
      </div>
      <textarea class="llm-input" rows="1" data-role="input" spellcheck="false" aria-label="Text to tokenize with Gemma 4"></textarea>
      <div class="llm-panel llm-scanner-wrap" data-role="stream-panel">
        <div class="llm-token-stream" data-role="stream"></div>
      </div>
      <div class="llm-panel" data-role="inspector" style="margin-bottom:0"></div>
      <div class="llm-stats" data-role="stats"></div>
    `;

    const statusBadgeEl = el.querySelector('[data-role="engine-status"]');
    const presetsEl = el.querySelector('[data-role="presets"]');
    const inputEl = el.querySelector('[data-role="input"]');
    const streamPanelEl = el.querySelector('[data-role="stream-panel"]');
    const streamEl = el.querySelector('[data-role="stream"]');
    const inspectorEl = el.querySelector('[data-role="inspector"]');
    const statsEl = el.querySelector('[data-role="stats"]');

    const syncTextareaRows = () => {
      const lines = currentText.split("\n").length;
      inputEl.rows = Math.min(4, Math.max(1, lines));
    };

    const syncStatusBadge = ({ status, statusLabel }) => {
      if (!statusBadgeEl) return;
      statusBadgeEl.textContent = statusLabel;
      statusBadgeEl.classList.remove(
        "llm-widget__badge--live",
        "llm-widget__badge--loading",
        "llm-widget__badge--warn"
      );
      if (status === "ready") {
        statusBadgeEl.classList.add("llm-widget__badge--live");
        statusBadgeEl.removeAttribute("role");
        statusBadgeEl.removeAttribute("tabindex");
      } else if (status === "error") {
        statusBadgeEl.classList.add("llm-widget__badge--warn");
        statusBadgeEl.setAttribute("role", "button");
        statusBadgeEl.setAttribute("tabindex", "0");
      } else {
        statusBadgeEl.classList.add("llm-widget__badge--loading");
        statusBadgeEl.removeAttribute("role");
        statusBadgeEl.removeAttribute("tabindex");
      }
    };

    const renderView = () => {
      const tokens = currentTokens;
      if (selectedIdx >= tokens.length) selectedIdx = Math.max(0, tokens.length - 1);
      if (pinnedIdx !== null && pinnedIdx >= tokens.length) pinnedIdx = null;

      if (!presetsEl.children.length) {
        presetsEl.innerHTML = PRESET_TOKENIZATIONS.map(
          (p, i) => `<button type="button" class="llm-scenario-btn" data-preset="${i}">${escapeHtml(p.label)}</button>`
        ).join("");
      }
      presetsEl.querySelectorAll("[data-preset]").forEach((btn) => {
        const p = PRESET_TOKENIZATIONS[Number(btn.dataset.preset)];
        const isAct = Boolean(p && p.text === currentText);
        btn.classList.toggle("llm-scenario-btn--active", isAct);
        btn.setAttribute("aria-pressed", String(isAct));
      });

      const oldScanner = streamPanelEl.querySelector(".llm-scanner-bar");
      if (oldScanner) oldScanner.remove();
      if (showScanner) {
        const bar = document.createElement("div");
        bar.className = "llm-scanner-bar";
        streamPanelEl.appendChild(bar);
      }

      if (renderedTokensRef !== tokens || animateEntrance) {
        renderedTokensRef = tokens;
        streamEl.innerHTML = tokens
          .map(
            (tok, idx) => `
            <span class="llm-chip llm-chip--${tok.cat} ${animateEntrance ? "llm-chip--anim" : ""}" style="--chip-index:${idx}" data-idx="${idx}" tabindex="0" role="button">
              <span>${escapeHtml(tok.t)}</span>
              <span class="llm-chip__id">${tok.verified === false ? "~#" : "#"}${tok.id}</span>
            </span>`
          )
          .join("");
      }

      const hasPinned = pinnedIdx !== null && pinnedIdx < tokens.length;
      streamEl.querySelectorAll("[data-idx]").forEach((chipEl) => {
        const idx = Number(chipEl.dataset.idx);
        const isSelected = hasPinned && idx === pinnedIdx;
        const isDimmed = hasPinned && idx !== pinnedIdx;
        if (hasPinned) chipEl.classList.remove("llm-chip--anim");
        chipEl.classList.toggle("llm-chip--selected", isSelected);
        chipEl.classList.toggle("llm-chip--dimmed", isDimmed);
        chipEl.setAttribute("aria-pressed", String(isSelected));
      });

      const active = tokens[hasPinned ? pinnedIdx : selectedIdx];
      if (active) {
        const catMeta = TOKEN_CAT_META[active.cat] || TOKEN_CAT_META.word;
        const isVerified = active.verified !== false;
        const vocabPct = Math.min(98, Math.max(2, Math.round((active.id / 262144) * 100)));
        const svgTokLabel = active.t.length > 15 ? `${active.t.slice(0, 14)}…` : active.t;
        inspectorEl.innerHTML = `
          <div style="display:flex;flex-wrap:wrap;align-items:center;gap:0.45rem;font-size:0.82rem;">
            <code style="background:${catMeta.bg};color:${catMeta.fg};border:1px solid ${catMeta.border};padding:0.08rem 0.42rem;border-radius:var(--gemma-radius-sm);font-weight:700;">${escapeHtml(active.t)}</code>
            <span class="llm-widget__badge" style="background:${catMeta.bg};border-color:${catMeta.border};color:${catMeta.fg};">${catMeta.name}</span>
            <span style="color:var(--gemma-text-secondary)">— ${escapeHtml(active.note)}</span>
          </div>
          <svg viewBox="0 0 420 26" style="width:100%;height:26px;display:block;margin-top:0.35rem;" aria-label="Vocabulary lookup beam">
            <rect x="2" y="5" width="96" height="16" rx="3" fill="${catMeta.bg}" stroke="${catMeta.fg}" />
            <text x="50" y="16" text-anchor="middle" font-size="9.5" font-family="monospace" font-weight="700" fill="${catMeta.fg}">${escapeHtml(svgTokLabel)}</text>
            <line x1="98" y1="13" x2="${125 + vocabPct * 1.6}" y2="13" stroke="${catMeta.fg}" stroke-width="2" class="llm-beam-line" />
            <rect x="125" y="5" width="290" height="16" rx="3" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <rect x="${125 + vocabPct * 2.6}" y="5" width="28" height="16" rx="2" fill="${catMeta.bg}" stroke="${catMeta.fg}" stroke-width="1.2" />
            <text x="270" y="16" text-anchor="middle" font-size="9" font-family="monospace" fill="var(--gemma-text-secondary)">262,144 Vocab Row ${isVerified ? "#" : "~#"}${active.id}</text>
          </svg>
        `;
      } else {
        inspectorEl.innerHTML = `<span style="font-size:0.82rem;color:var(--gemma-text-muted)">Type above to inspect real Gemma 4 tokens.</span>`;
      }

      const charCount = currentText.length;
      const tokCount = tokens.length;
      const ratioNum = tokCount ? charCount / tokCount : 0;
      statsEl.innerHTML = `
        <div class="llm-stat">Characters<strong data-stat="chars">${charCount}</strong></div>
        <div class="llm-stat">Gemma 4 Tokens<strong data-stat="toks">${tokCount}</strong></div>
        <div class="llm-stat">Chars / Token<strong data-stat="ratio">${ratioNum.toFixed(2)}</strong></div>
        <div class="llm-stat">Vocabulary Size<strong>262,144</strong></div>
      `;

      if (animateEntrance) {
        animateTicker(statsEl.querySelector('[data-stat="chars"]'), 0, charCount, 420, (v) => String(Math.round(v)));
        animateTicker(statsEl.querySelector('[data-stat="toks"]'), 0, tokCount, 420, (v) => String(Math.round(v)));
        animateTicker(statsEl.querySelector('[data-stat="ratio"]'), 0, ratioNum, 480, (v) => v.toFixed(2));
      }
      animateEntrance = false;
      showScanner = false;
    };

    const runTokenize = async () => {
      const seq = ++encodeSeq;
      const textSnap = currentText;
      const tokens = await realTokenizerService.encode(textSnap);
      if (seq !== encodeSeq) return;
      currentTokens = tokens;
      renderView();
    };

    inputEl.value = currentText;
    syncTextareaRows();
    syncStatusBadge(realTokenizerService.getState());

    const unsubscribe = realTokenizerService.subscribe((st) => {
      if (!el.isConnected) {
        unsubscribe();
        return;
      }
      syncStatusBadge(st);
      if (st.status === "ready") {
        runTokenize();
      }
    });

    const startBackgroundLoad = () => {
      realTokenizerService.ensureReady();
    };
    if (typeof window !== "undefined" && typeof window.requestIdleCallback === "function") {
      window.requestIdleCallback(startBackgroundLoad, { timeout: 600 });
    } else {
      setTimeout(startBackgroundLoad, 120);
    }

    inputEl.addEventListener("input", () => {
      currentText = inputEl.value;
      syncTextareaRows();
      selectedIdx = 0;
      pinnedIdx = null;
      animateEntrance = false;
      runTokenize();
    });

    el.addEventListener("click", (e) => {
      const retryBadge = e.target.closest('[data-role="engine-status"]');
      if (retryBadge && realTokenizerService.getState().status === "error") {
        realTokenizerService.retry();
        return;
      }
      const btn = e.target.closest("[data-preset]");
      if (btn) {
        const p = PRESET_TOKENIZATIONS[Number(btn.dataset.preset)];
        if (!p) return;
        currentText = p.text;
        inputEl.value = currentText;
        syncTextareaRows();
        selectedIdx = 0;
        pinnedIdx = null;
        animateEntrance = true;
        showScanner = true;
        runTokenize();
        return;
      }
      const chip = e.target.closest("[data-idx]");
      if (chip) {
        const clickedIdx = Number(chip.dataset.idx);
        if (pinnedIdx === clickedIdx) {
          pinnedIdx = null;
        } else {
          pinnedIdx = clickedIdx;
          selectedIdx = clickedIdx;
        }
        renderView();
        return;
      }
      if (pinnedIdx !== null) {
        pinnedIdx = null;
        renderView();
      }
    });

    document.addEventListener("click", (e) => {
      if (!el.isConnected || pinnedIdx === null) return;
      if (!el.contains(e.target)) {
        pinnedIdx = null;
        renderView();
      }
    });

    streamEl.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && pinnedIdx !== null) {
        pinnedIdx = null;
        renderView();
        return;
      }
      if (e.key !== "Enter" && e.key !== " ") return;
      const chip = e.target.closest("[data-idx]");
      if (!chip) return;
      e.preventDefault();
      const clickedIdx = Number(chip.dataset.idx);
      if (pinnedIdx === clickedIdx) {
        pinnedIdx = null;
      } else {
        pinnedIdx = clickedIdx;
        selectedIdx = clickedIdx;
      }
      renderView();
    });

    streamEl.addEventListener("mouseover", (e) => {
      if (pinnedIdx !== null) return;
      const chip = e.target.closest("[data-idx]");
      if (!chip) return;
      const nextIdx = Number(chip.dataset.idx);
      if (nextIdx !== selectedIdx) {
        selectedIdx = nextIdx;
        renderView();
      }
    });

    streamEl.addEventListener("animationend", (e) => {
      if (e.animationName !== "llmBadgeFly") return;
      const chip = e.target.closest(".llm-chip--anim");
      if (chip) chip.classList.remove("llm-chip--anim");
    });

    renderView();
    if (realTokenizerService.getState().status === "ready") {
      runTokenize();
    }
  };

  /* =====================================================================
     2. EMBEDDING SPACE & PLE EXPLORER (Step 2)
     Verified Gemma 4 E2B token IDs & first-8 BF16 embed_tokens.weight
     values + illustrative 2D semantic projection & PLE circuit visuals.
     ===================================================================== */
  const EMBED_POINTS = [
    // Animals & Multimodal (Dims 0-2 active: cos(cat,kitten)=0.43, cos(kitten,puppy)=0.54, cos(kitten-cat+dog,puppy)=0.51)
    { label: "▁cat", shortLabel: "▁cat", id: 5866, x: 84, y: 76, cluster: "Animals & Multimodal", realVec8: [-0.0049, -0.0491, -0.0098, 0.0133, 0.0476, -0.0437, 0.0315, -0.0099], vec: [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000] },
    { label: "▁kitten", shortLabel: "▁kitten", id: 55095, x: 146, y: 46, cluster: "Animals & Multimodal", realVec8: [0.0801, -0.0488, -0.0084, 0.0308, 0.0193, -0.0403, -0.0053, -0.0043], vec: [0.4300, 0.9028, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000] },
    { label: "▁dog", shortLabel: "▁dog", id: 4799, x: 90, y: 118, cluster: "Animals & Multimodal", realVec8: [-0.0059, -0.0237, -0.0186, -0.0074, -0.0176, 0.0184, 0.0466, -0.0898], vec: [0.5200, 0.0736, 0.8510, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000] },
    { label: "▁puppy", shortLabel: "▁puppy", id: 34646, x: 152, y: 88, cluster: "Animals & Multimodal", realVec8: [0.0576, -0.0091, 0.0104, -0.0002, 0.0371, 0.0347, 0.0537, -0.0063], vec: [0.3300, 0.4409, 0.2200, 0.0000, 0.0000, 0.0000, 0.8054, 0.0000] },
    { label: "🖼 [Image: cat]", shortLabel: "🖼 Image", id: 258880, x: 22, y: 46, cluster: "Animals & Multimodal", isSoftToken: true, realVec8: [-0.0018, -0.0029, 0.0596, -0.0439, 0.0171, 0.0055, 0.0093, 0.0309], vec: [0.6800, 0.3400, 0.0800, 0.0000, 0.0000, 0.0000, 0.0000, 0.6447] },
    { label: "🔊 [Audio: meow]", shortLabel: "🔊 Audio", id: 258881, x: 22, y: 108, cluster: "Animals & Multimodal", isSoftToken: true, realVec8: [-0.0003, -0.0015, 0.0569, -0.0459, 0.0192, 0.0053, 0.0082, 0.0284], vec: [0.6400, 0.3100, 0.1200, 0.0000, 0.0000, 0.0000, 0.0000, -0.6927] },
    // Royalty & People (Dims 3-5 active: cos(king,queen)=0.58, cos(man,woman)=0.60, cos(king-man+woman,queen)=0.56)
    { label: "▁king", shortLabel: "▁king", id: 9615, x: 244, y: 52, cluster: "Royalty & People", realVec8: [-0.0199, -0.0092, 0.0525, -0.0361, -0.0054, 0.0488, 0.0325, -0.0262], vec: [0.0000, 0.0000, 0.0000, 0.7200, 0.4583, -0.5212, 0.0000, 0.0000] },
    { label: "▁queen", shortLabel: "▁queen", id: 26476, x: 332, y: 48, cluster: "Royalty & People", realVec8: [-0.0147, 0.0347, 0.0405, -0.0193, 0.0605, 0.0154, 0.0022, -0.0013], vec: [0.0000, 0.0000, 0.0000, 0.7200, -0.4583, -0.5212, 0.0000, 0.0000] },
    { label: "▁man", shortLabel: "▁man", id: 880, x: 230, y: 112, cluster: "Royalty & People", realVec8: [0.0063, 0.0126, 0.0070, 0.0072, 0.0002, -0.0447, 0.0074, 0.0073], vec: [0.0000, 0.0000, 0.0000, 0.7500, 0.4472, 0.4873, 0.0000, 0.0000] },
    { label: "▁woman", shortLabel: "▁woman", id: 3875, x: 318, y: 108, cluster: "Royalty & People", realVec8: [0.0008, 0.0130, 0.0059, 0.0220, 0.0161, -0.0282, 0.0060, -0.0086], vec: [0.0000, 0.0000, 0.0000, 0.7500, -0.4472, 0.4873, 0.0000, 0.0000] },
    // Tech & AI
    { label: "▁Gemma", shortLabel: "▁Gemma", id: 147224, x: 238, y: 176, cluster: "Tech & AI", realVec8: [-0.0087, 0.0037, 0.0039, 0.0311, -0.0398, 0.0220, 0.0200, 0.0298], vec: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.7800, 0.6258] },
    { label: "▁model", shortLabel: "▁model", id: 2028, x: 328, y: 216, cluster: "Tech & AI", realVec8: [0.0041, 0.0074, 0.0542, 0.0669, 0.0007, 0.0273, 0.0115, -0.0281], vec: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2800, 0.6800, 0.6776] },
    { label: "▁token", shortLabel: "▁token", id: 8369, x: 230, y: 218, cluster: "Tech & AI", realVec8: [0.0154, -0.0182, 0.0469, -0.0294, -0.0021, -0.0393, 0.0305, -0.0513], vec: [0.0000, 0.0000, 0.0000, 0.0000, 0.2500, 0.0000, 0.6400, 0.7266] },
    { label: "▁neural", shortLabel: "▁neural", id: 22823, x: 326, y: 170, cluster: "Tech & AI", realVec8: [-0.0222, 0.0043, 0.0276, -0.0283, -0.0188, -0.0186, 0.0239, 0.0210], vec: [0.0000, 0.0000, 0.0000, 0.0000, -0.2500, 0.0000, 0.7100, 0.6583] },
    // Actions & Code
    { label: "▁run", shortLabel: "▁run", id: 1845, x: 46, y: 174, cluster: "Actions & Code", realVec8: [0.0062, -0.0062, 0.0128, -0.0087, 0.0292, -0.0214, 0.0552, 0.0001], vec: [0.0000, 0.0000, 0.1800, 0.0000, 0.0000, 0.0000, -0.7600, 0.6245] },
    { label: "▁sprint", shortLabel: "▁sprint", id: 29991, x: 34, y: 214, cluster: "Actions & Code", realVec8: [0.0767, -0.0620, -0.0143, -0.0854, 0.0110, -0.0267, 0.0723, 0.0309], vec: [0.0000, 0.1500, 0.0000, 0.0000, 0.0000, 0.0000, -0.6800, 0.7177] },
    { label: "▁python", shortLabel: "▁python", id: 23181, x: 122, y: 194, cluster: "Actions & Code", realVec8: [0.0011, 0.0532, -0.0038, -0.0461, -0.0085, 0.0535, -0.0206, 0.0250], vec: [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.3200, 0.9474] },
  ];

  const cosineSim = (a, b) => {
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  };

  const ANALOGY_CONFIGS = {
    royalty: {
      hudFormula: "vec(▁king) − vec(▁man) + vec(▁woman) ≈ vec(▁queen)",
      idxA: 6, // ▁king (244, 52)
      idxB: 8, // ▁man (230, 112)
      idxC: 9, // ▁woman (318, 108)
      idxD: 7, // ▁queen (332, 48)
      activeSet: [6, 7, 8, 9],
      deltaLabel: "Δ Royalty (+14, −60)",
      sim1536d: "0.56",
      steps: ["1. Δ = A − B", "2. Shift Δ → C", "3. C + Δ ≈ D"],
    },
    pets: {
      hudFormula: "vec(▁kitten) − vec(▁cat) + vec(▁dog) ≈ vec(▁puppy)",
      idxA: 1, // ▁kitten (146, 46)
      idxB: 0, // ▁cat (84, 76)
      idxC: 2, // ▁dog (90, 118)
      idxD: 3, // ▁puppy (152, 88)
      activeSet: [0, 1, 2, 3],
      deltaLabel: "Δ Youth (+62, −30)",
      sim1536d: "0.51",
      steps: ["1. Δ = A − B", "2. Shift Δ → C", "3. C + Δ ≈ D"],
    },
    multimodal: {
      hudFormula: "Proj(🖼 #258880) ≈ Proj(🔊 #258881) ≈ vec(▁cat #5866)",
      idxA: 4, // 🖼 Image
      idxB: 5, // 🔊 Audio
      idxC: 0, // ▁cat
      idxD: 0, // ▁cat
      activeSet: [0, 4, 5],
      sim1536d: "1,536-d",
      steps: ["1. Vision Proj", "2. Audio Proj", "3. Shared Space"],
    },
  };

  const PLE_MODELS = [
    {
      id: "e2b",
      pill: "E2B · 35L PLE",
      shortTag: "E2B",
      fullName: "Gemma 4 E2B",
      hasPle: true,
      layers: 35,
      dModel: 1536,
      dModelFmt: "1,536",
      dPle: 256,
      activeB: 2.37,
      embB: 0.40,
      pleB: 2.35,
      sparseB: 2.75,
      totalB: 5.12,
    },
    {
      id: "e4b",
      pill: "E4B · 42L PLE",
      shortTag: "E4B",
      fullName: "Gemma 4 E4B",
      hasPle: true,
      layers: 42,
      dModel: 2560,
      dModelFmt: "2,560",
      dPle: 256,
      activeB: 4.51,
      embB: 0.67,
      pleB: 2.82,
      sparseB: 3.49,
      totalB: 8.00,
    },
    {
      id: "12b",
      pill: "12B · No PLE",
      shortTag: "12B",
      fullName: "Gemma 4 12B Unified",
      hasPle: false,
      layers: 48,
      dModel: 3840,
      dModelFmt: "3,840",
      dPle: 0,
      activeB: 11.95,
      embB: 1.01,
      pleB: 0,
      sparseB: 0,
      totalB: 11.95,
    },
    {
      id: "26b",
      pill: "26B MoE · No PLE",
      shortTag: "26B",
      fullName: "Gemma 4 26B A4B MoE",
      hasPle: false,
      layers: 30,
      dModel: 2816,
      dModelFmt: "2,816",
      dPle: 0,
      activeB: 25.20,
      embB: 0.74,
      pleB: 0,
      sparseB: 0,
      totalB: 25.20,
    },
    {
      id: "31b",
      pill: "31B · No PLE",
      shortTag: "31B",
      fullName: "Gemma 4 31B Dense",
      hasPle: false,
      layers: 60,
      dModel: 5376,
      dModelFmt: "5,376",
      dPle: 0,
      activeB: 30.70,
      embB: 1.41,
      pleB: 0,
      sparseB: 0,
      totalB: 30.70,
    },
  ];

  const PLE_SAMPLE_IDXS = [0, 6, 16, 4]; // ▁cat, ▁king, ▁python, 🖼 Image

  const SPACE_TOUR_ORDER = [0, 1, 4, 5, 6, 7, 16, 12];

  const mountEmbeddings = (el) => {
    let tab = "space"; // "space" | "ple"
    let activeIdx = 0;
    let analogyMode = null; // null | "royalty" | "pets" | "multimodal"
    let analogyStep = 0; // 0 | 1 | 2
    let analogyTimer = null;
    let spaceTourTimer = null;
    let slideStartMs = 0;
    let pleModIdx = 0;
    let selectedPleLayer = 18;
    let pleSubphase = -1; // -1 idle, 0 Flash slice drop, 1 Gate & Residual merge
    let pleCascadeTimer = null;
    let rafId = 0;
    let polyParticles = [];
    let flowWires = [];
    let sonarRings = [];
    let burstRings = [];
    let translateGroups = [];

    const panelEl = el.closest("[data-llm-panel]");
    const shellEl = el.closest("[data-llm-shell]");
    const isVisible = () => el.isConnected && (!panelEl || !panelEl.hidden);

    el.innerHTML = `
      <div class="llm-widget__header">
        <p class="llm-widget__title">Semantic Vector Space &amp; Per-Layer Embeddings (PLE)</p>
        <div class="llm-toolbar__group" role="group" aria-label="Embeddings view mode">
          <span class="llm-toolbar__label">View</span>
          <div class="llm-seg">
            <button type="button" class="llm-seg__btn llm-seg__btn--active" data-emb-tab="space">Tab A: Lookup &amp; Vector Space</button>
            <button type="button" class="llm-seg__btn" data-emb-tab="ple">Tab B: Per-Layer Embeddings (PLE)</button>
          </div>
        </div>
      </div>
      <div data-role="body"></div>
    `;

    const bodyEl = el.querySelector('[data-role="body"]');

    const nowMs = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

    const stopTimers = () => {
      if (analogyTimer) {
        clearInterval(analogyTimer);
        analogyTimer = null;
      }
      if (spaceTourTimer) {
        clearInterval(spaceTourTimer);
        spaceTourTimer = null;
      }
      if (pleCascadeTimer) {
        clearInterval(pleCascadeTimer);
        pleCascadeTimer = null;
      }
      pleSubphase = -1;
    };

    const cacheAnimNodes = () => {
      polyParticles = Array.from(bodyEl.querySelectorAll("[data-anim-pts]")).map((node) => {
        const pts = node.dataset.animPts.split(";").map((p) => p.split(",").map(Number));
        const segs = [];
        let total = 0;
        for (let i = 0; i < pts.length - 1; i++) {
          const dx = pts[i + 1][0] - pts[i][0];
          const dy = pts[i + 1][1] - pts[i][1];
          const len = Math.hypot(dx, dy);
          segs.push({ x0: pts[i][0], y0: pts[i][1], dx, dy, len });
          total += len;
        }
        return {
          node,
          segs,
          total: total || 1,
          dur: Number(node.dataset.dur) || 1000,
          phase: Number(node.dataset.phase) || 0,
        };
      });

      flowWires = Array.from(bodyEl.querySelectorAll("[data-flow-wire]")).map((node) => ({
        node,
        speed: Number(node.dataset.flowWire) || 24,
      }));

      sonarRings = Array.from(bodyEl.querySelectorAll("[data-sonar-ring]")).map((node) => ({
        node,
        phase: Number(node.dataset.sonarRing) || 0,
      }));

      burstRings = Array.from(bodyEl.querySelectorAll("[data-burst-ring]"));

      translateGroups = Array.from(bodyEl.querySelectorAll("[data-anim-translate]")).map((node) => {
        const [dx, dy] = node.dataset.animTranslate.split(",").map(Number);
        return {
          node,
          dx: dx || 0,
          dy: dy || 0,
          dur: Number(node.dataset.dur) || 850,
        };
      });
    };

    const stepParticles = (now) => {
      for (let i = 0; i < polyParticles.length; i++) {
        const p = polyParticles[i];
        const t = ((now / p.dur + p.phase) % 1 + 1) % 1;
        let dist = t * p.total;
        let px = p.segs[0] ? p.segs[0].x0 : 0;
        let py = p.segs[0] ? p.segs[0].y0 : 0;
        for (let s = 0; s < p.segs.length; s++) {
          const seg = p.segs[s];
          if (dist <= seg.len || s === p.segs.length - 1) {
            const u = seg.len > 0 ? dist / seg.len : 0;
            px = seg.x0 + seg.dx * u;
            py = seg.y0 + seg.dy * u;
            break;
          }
          dist -= seg.len;
        }
        p.node.setAttribute("cx", px.toFixed(1));
        p.node.setAttribute("cy", py.toFixed(1));
      }
      for (let i = 0; i < flowWires.length; i++) {
        const fw = flowWires[i];
        const offset = -((now / fw.speed) % 24);
        fw.node.setAttribute("stroke-dashoffset", offset.toFixed(1));
      }
      for (let i = 0; i < sonarRings.length; i++) {
        const sr = sonarRings[i];
        const u = ((now / 2100 + sr.phase) % 1 + 1) % 1;
        const r = 6 + u * 20;
        const op = (1 - u) * 0.65;
        sr.node.setAttribute("r", r.toFixed(1));
        sr.node.setAttribute("stroke-opacity", op.toFixed(2));
      }
      for (let i = 0; i < burstRings.length; i++) {
        const br = burstRings[i];
        const u = ((now / 1300) % 1 + 1) % 1;
        const r = 9 + Math.sin(u * Math.PI) * 6.5;
        const op = 0.82 - u * 0.45;
        br.setAttribute("r", r.toFixed(1));
        br.setAttribute("stroke-opacity", op.toFixed(2));
      }
      for (let i = 0; i < translateGroups.length; i++) {
        const tg = translateGroups[i];
        const elapsed = Math.max(0, now - slideStartMs);
        const rawT = Math.min(1, elapsed / tg.dur);
        const eased = 1 - Math.pow(1 - rawT, 3);
        const tx = tg.dx * eased;
        const ty = tg.dy * eased;
        tg.node.setAttribute("transform", `translate(${tx.toFixed(1)}, ${ty.toFixed(1)})`);
      }
    };

    const ensureRaf = () => {
      if (rafId || typeof window.requestAnimationFrame !== "function") return;
      if (!isVisible()) return;
      const tick = (now) => {
        if (!isVisible()) {
          rafId = 0;
          return;
        }
        stepParticles(now);
        rafId = window.requestAnimationFrame(tick);
      };
      rafId = window.requestAnimationFrame(tick);
    };

    if (panelEl && typeof MutationObserver === "function") {
      const obs = new MutationObserver(() => {
        if (isVisible()) ensureRaf();
      });
      obs.observe(panelEl, { attributes: true, attributeFilter: ["hidden"] });
    }
    if (shellEl) {
      shellEl.addEventListener("llm:step", () => {
        if (isVisible()) ensureRaf();
      });
    }

    const syncAnalogyTargetIdx = () => {
      if (!analogyMode || !ANALOGY_CONFIGS[analogyMode]) return;
      const cfg = ANALOGY_CONFIGS[analogyMode];
      activeIdx = analogyStep === 0 ? cfg.idxA : analogyStep === 1 ? cfg.idxC : cfg.idxD;
    };

    const startAnalogyAnim = () => {
      stopTimers();
      analogyStep = 0;
      syncAnalogyTargetIdx();
      slideStartMs = nowMs();
      analogyTimer = setInterval(() => {
        if (!el.isConnected) {
          stopTimers();
          return;
        }
        if (analogyStep < 2) {
          analogyStep += 1;
          if (analogyStep === 1) slideStartMs = nowMs();
          syncAnalogyTargetIdx();
          render();
        } else {
          stopTimers();
          render();
        }
      }, 1250);
      render();
    };

    const renderVectorArrow = (x1, y1, x2, y2, color, dashed = false, opacity = 1) => {
      const dx = x2 - x1;
      const dy = y2 - y1;
      const len = Math.hypot(dx, dy) || 1;
      const padEnd = 6.5;
      const padStart = 4.5;
      const sx = x1 + (dx / len) * padStart;
      const sy = y1 + (dy / len) * padStart;
      const ex = x2 - (dx / len) * padEnd;
      const ey = y2 - (dy / len) * padEnd;
      const angle = Math.atan2(dy, dx) * (180 / Math.PI);
      return `
        <g opacity="${opacity}">
          <line x1="${sx.toFixed(1)}" y1="${sy.toFixed(1)}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}"
            stroke="${color}" stroke-width="2.3" ${dashed ? 'stroke-dasharray="4 2.5"' : ""} stroke-linecap="round" />
          <polygon points="0,0 -7,-3.5 -7,3.5" fill="${color}"
            transform="translate(${ex.toFixed(1)},${ey.toFixed(1)}) rotate(${angle.toFixed(1)})" />
        </g>
      `;
    };

    const computePleChannelVal = (tokIdx, layerNum, totalLayers, binIdx, isSoftToken) => {
      const progress = (layerNum - 1) / Math.max(1, totalLayers - 1); // 0..1
      const bandCenter = progress < 0.3 ? 3.5 : progress < 0.68 ? 11.5 : 19.5;
      const dist = Math.abs(binIdx - bandCenter);
      const envelope = Math.max(0.16, Math.exp(-(dist * dist) / 26));
      const seed = (tokIdx + 1) * 17 + layerNum * 7 + binIdx * 13;
      const wave = isSoftToken
        ? Math.sin(binIdx * 0.72 + layerNum * 0.35) * 0.85
        : Math.sin(seed * 0.63) * 0.65 + Math.cos(seed * 0.29) * 0.35;
      return Math.max(-1, Math.min(1, wave * envelope * 1.25));
    };

    const renderSpaceTab = () => {
      const target = EMBED_POINTS[activeIdx];
      const ranked = EMBED_POINTS.map((pt, idx) => ({
        ...pt,
        idx,
        sim: cosineSim(target.vec, pt.vec),
      }))
        .filter((pt) => pt.idx !== activeIdx)
        .sort((a, b) => b.sim - a.sim)
        .slice(0, 5);

      const gridLines = [];
      for (let gx = 52; gx < 420; gx += 52) {
        gridLines.push(`<line x1="${gx}" y1="8" x2="${gx}" y2="256" stroke="var(--gemma-border)" stroke-width="0.6" stroke-opacity="0.45" />`);
      }
      for (let gy = 44; gy < 264; gy += 44) {
        gridLines.push(`<line x1="8" y1="${gy}" x2="412" y2="${gy}" stroke="var(--gemma-border)" stroke-width="0.6" stroke-opacity="0.45" />`);
      }

      const raysSvg = analogyMode
        ? ""
        : ranked
            .slice(0, 3)
            .map((nb, rIdx) => {
              const mx = (target.x + nb.x) / 2;
              const my = (target.y + nb.y) / 2;
              return `
                <line x1="${target.x}" y1="${target.y}" x2="${nb.x}" y2="${nb.y}"
                  stroke="var(--gemma-accent)" stroke-width="${rIdx === 0 ? "1.8" : "1.25"}" stroke-opacity="${rIdx === 0 ? "0.75" : "0.48"}" class="llm-measure-ray" />
                <circle r="${rIdx === 0 ? "2.6" : "2.1"}" fill="var(--gemma-accent-fill)"
                  data-anim-pts="${target.x},${target.y};${nb.x},${nb.y}" data-dur="${820 + rIdx * 170}" data-phase="${(rIdx * 0.31).toFixed(2)}" />
                <g pointer-events="none">
                  <rect x="${mx - 14}" y="${my - 6.5}" width="28" height="12" rx="6"
                    fill="var(--gemma-surface)" stroke="var(--gemma-accent)" stroke-width="0.9" />
                  <text x="${mx}" y="${my + 2.2}" text-anchor="middle" font-size="7.6" font-family="monospace"
                    fill="var(--gemma-accent)" font-weight="700">${nb.sim.toFixed(2)}</text>
                </g>
              `;
            })
            .join("");

      const sonarSvg = analogyMode
        ? ""
        : `
          <circle cx="${target.x}" cy="${target.y}" r="8" fill="none" stroke="var(--gemma-accent)" stroke-width="1.5" stroke-opacity="0.55" data-sonar-ring="0" />
          <circle cx="${target.x}" cy="${target.y}" r="8" fill="none" stroke="var(--gemma-accent)" stroke-width="1.2" stroke-opacity="0.35" data-sonar-ring="0.5" />
        `;

      let overlaySvg = "";
      if (analogyMode === "royalty" || analogyMode === "pets") {
        const cfg = ANALOGY_CONFIGS[analogyMode];
        const ptA = EMBED_POINTS[cfg.idxA];
        const ptB = EMBED_POINTS[cfg.idxB];
        const ptC = EMBED_POINTS[cfg.idxC];
        const ptD = EMBED_POINTS[cfg.idxD];
        const shiftX = ptC.x - ptB.x;
        const shiftY = ptC.y - ptB.y;
        const isRoyalty = analogyMode === "royalty";
        const deltaBoxX = isRoyalty ? 222 : 52;
        const deltaBoxY = 23;

        const polyFill =
          analogyStep >= 1
            ? `<polygon points="${ptB.x},${ptB.y} ${ptA.x},${ptA.y} ${ptD.x},${ptD.y} ${ptC.x},${ptC.y}"
                fill="var(--gemma-accent)" fill-opacity="0.09" stroke="var(--gemma-border-strong)" stroke-width="1" stroke-dasharray="2.5 2.5" />`
            : "";

        const arrowBA = renderVectorArrow(ptB.x, ptB.y, ptA.x, ptA.y, "var(--gemma-status-warning)", false, 1);
        const particleBA = `<circle r="2.8" fill="var(--gemma-status-warning)" data-anim-pts="${ptB.x},${ptB.y};${ptA.x},${ptA.y}" data-dur="780" data-phase="0" />`;

        const slidingArrow =
          analogyStep === 1
            ? `
              <g data-anim-translate="${shiftX},${shiftY}" data-dur="880">
                ${renderVectorArrow(ptB.x, ptB.y, ptA.x, ptA.y, "var(--gemma-status-success)", false, 0.94)}
              </g>
              ${renderVectorArrow(ptB.x, ptB.y, ptC.x, ptC.y, "var(--gemma-accent)", true, 0.65)}
              <circle r="2.6" fill="var(--gemma-accent-fill)" data-anim-pts="${ptB.x},${ptB.y};${ptC.x},${ptC.y}" data-dur="780" data-phase="0.2" />
            `
            : "";

        const dockedArrow =
          analogyStep >= 2
            ? `
              ${renderVectorArrow(ptB.x, ptB.y, ptC.x, ptC.y, "var(--gemma-accent)", true, 0.55)}
              ${renderVectorArrow(ptC.x, ptC.y, ptD.x, ptD.y, "var(--gemma-status-success)", false, 1)}
              <circle r="2.8" fill="var(--gemma-status-success)" data-anim-pts="${ptC.x},${ptC.y};${ptD.x},${ptD.y}" data-dur="780" data-phase="0.4" />
              <circle cx="${ptD.x}" cy="${ptD.y}" r="12" fill="none" stroke="var(--gemma-status-success)" stroke-width="1.8" stroke-opacity="0.7" data-burst-ring="1" />
            `
            : "";

        const roleBadges = [
          { pt: ptB, tag: "B", color: "var(--gemma-text-muted)" },
          { pt: ptA, tag: "A", color: "var(--gemma-status-warning)" },
          { pt: ptC, tag: "C", color: "var(--gemma-accent)" },
          { pt: ptD, tag: "D", color: "var(--gemma-status-success)" },
        ]
          .map(
            (b) => `
              <circle cx="${b.pt.x}" cy="${b.pt.y}" r="9" fill="none" stroke="${b.color}" stroke-width="1.4" stroke-dasharray="2 2" />
              <rect x="${b.pt.x - 15}" y="${b.pt.y - 14}" width="12" height="10" rx="2.5" fill="var(--gemma-surface)" stroke="${b.color}" stroke-width="0.9" />
              <text x="${b.pt.x - 9}" y="${b.pt.y - 6.8}" text-anchor="middle" font-size="6.8" font-weight="700" font-family="monospace" fill="${b.color}">${b.tag}</text>
            `
          )
          .join("");

        const hudText = `${cfg.hudFormula}  ·  1536d cos = ${cfg.sim1536d}`;
        overlaySvg = `
          ${polyFill}
          ${roleBadges}
          ${arrowBA}
          ${particleBA}
          ${slidingArrow}
          ${dockedArrow}
          <rect x="${deltaBoxX}" y="${deltaBoxY}" width="112" height="13" rx="3.5"
            fill="var(--gemma-surface)" stroke="var(--gemma-status-warning)" stroke-width="0.9" />
          <text x="${deltaBoxX + 56}" y="${deltaBoxY + 9.2}" text-anchor="middle" font-size="7.4"
            font-family="monospace" font-weight="700" fill="var(--gemma-status-warning)">${escapeHtml(cfg.deltaLabel)}</text>
          <rect x="44" y="238" width="332" height="16" rx="5" fill="var(--gemma-surface)" stroke="var(--gemma-cat-tool-border)" stroke-width="1" />
          <text x="210" y="248.8" text-anchor="middle" font-size="7.8" font-family="monospace" font-weight="700" fill="var(--gemma-cat-tool-fg)">${escapeHtml(hudText)}</text>
        `;
      } else if (analogyMode === "multimodal") {
        const cfg = ANALOGY_CONFIGS.multimodal;
        const c = EMBED_POINTS[0];
        const img = EMBED_POINTS[4];
        const aud = EMBED_POINTS[5];
        const showVision = analogyStep >= 0;
        const showAudio = analogyStep >= 1;
        const showConverge = analogyStep >= 2;
        overlaySvg = `
          <circle cx="${c.x}" cy="${c.y}" r="24" fill="var(--gemma-accent)" fill-opacity="0.08"
            stroke="var(--gemma-accent)" stroke-width="1.4" stroke-dasharray="3 2.5" />
          <!-- Mini image 2x2 patch grid near Image soft token -->
          <g transform="translate(13, 21)">
            <rect x="0" y="0" width="8" height="8" rx="1" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-accent)" stroke-width="0.8" />
            <rect x="9" y="0" width="8" height="8" rx="1" fill="var(--gemma-accent-fill)" fill-opacity="0.75" />
            <rect x="0" y="9" width="8" height="8" rx="1" fill="var(--gemma-accent-fill)" fill-opacity="0.45" />
            <rect x="9" y="9" width="8" height="8" rx="1" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-accent)" stroke-width="0.8" />
          </g>
          <!-- Mini audio waveform bars near Audio soft token -->
          <g transform="translate(14, 118)" stroke="var(--gemma-status-warning)" stroke-width="2" stroke-linecap="round">
            <line x1="2" y1="6" x2="2" y2="11" />
            <line x1="6" y1="2" x2="6" y2="14" />
            <line x1="10" y1="4" x2="10" y2="12" />
            <line x1="14" y1="1" x2="14" y2="15" />
            <line x1="18" y1="5" x2="18" y2="11" />
          </g>
          ${
            showVision
              ? `${renderVectorArrow(img.x, img.y, c.x, c.y, "var(--gemma-accent)", false, 0.95)}
                 <circle r="2.8" fill="var(--gemma-accent-fill)" data-anim-pts="${img.x},${img.y};${c.x},${c.y}" data-dur="760" data-phase="0" />`
              : ""
          }
          ${
            showAudio
              ? `${renderVectorArrow(aud.x, aud.y, c.x, c.y, "var(--gemma-status-warning)", false, 0.95)}
                 <circle r="2.8" fill="var(--gemma-status-warning)" data-anim-pts="${aud.x},${aud.y};${c.x},${c.y}" data-dur="760" data-phase="0.45" />`
              : ""
          }
          ${
            showConverge
              ? `<circle cx="${c.x}" cy="${c.y}" r="13" fill="none" stroke="var(--gemma-status-success)" stroke-width="1.8" stroke-opacity="0.75" data-burst-ring="1" />`
              : ""
          }
          <rect x="44" y="238" width="332" height="16" rx="5" fill="var(--gemma-surface)" stroke="var(--gemma-cat-multi-border)" stroke-width="1" />
          <text x="210" y="248.8" text-anchor="middle" font-size="7.6" font-family="monospace" font-weight="700" fill="var(--gemma-cat-multi-fg)">${escapeHtml(cfg.hudFormula)}</text>
        `;
      }

      const modeKey = analogyMode && ANALOGY_CONFIGS[analogyMode] ? analogyMode : "free";

      // Right card 1: Visual O(1) Row Lookup + 8-dim BF16 Diverging Heatmap
      const activeRowSlot = activeIdx % 5;
      const matrixRowsSvg = Array.from({ length: 5 }, (_, r) => {
        const ry = 26 + r * 6;
        const isRow = r === activeRowSlot;
        return `<rect x="116" y="${ry}" width="84" height="4.2" rx="1.5"
          fill="${isRow ? "var(--gemma-accent-fill)" : "var(--gemma-surface-sunken)"}"
          stroke="${isRow ? "var(--gemma-accent)" : "var(--gemma-border)"}" stroke-width="0.6" />`;
      }).join("");

      const lookupMiddleSvg = target.isSoftToken
        ? `
          <rect x="110" y="22" width="96" height="34" rx="5" fill="var(--gemma-cat-multi-bg)" stroke="var(--gemma-cat-multi-border)" stroke-width="1.2" stroke-dasharray="3 2" />
          <text x="158" y="36" text-anchor="middle" font-size="7.6" font-weight="700" font-family="monospace" fill="var(--gemma-cat-multi-fg)">${target.id === 258880 ? "vision_proj" : "audio_proj"}</text>
          <text x="158" y="48" text-anchor="middle" font-size="6.8" font-family="monospace" fill="var(--gemma-text-secondary)">runtime → 1,536d</text>
        `
        : `
          <rect x="110" y="21" width="96" height="36" rx="4" fill="var(--gemma-surface)" stroke="var(--gemma-border-strong)" />
          ${matrixRowsSvg}
        `;

      const heatmapCellsSvg = target.realVec8
        .map((v, i) => {
          const cx = 8 + i * 33;
          const mag = Math.min(1, Math.abs(v) / 0.085);
          const bh = Math.max(2, Math.round(mag * 11));
          const isPos = v >= 0;
          const barY = isPos ? 88 - bh : 88;
          const fillCol = isPos ? "var(--gemma-accent-fill)" : "var(--gemma-status-warning)";
          const bgCol = isPos ? "var(--gemma-cat-turn-bg)" : "var(--gemma-cat-think-bg)";
          const fmt = (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3).slice(1);
          return `
            <rect x="${cx}" y="75" width="30" height="26" rx="3" fill="${bgCol}" fill-opacity="${(0.22 + mag * 0.55).toFixed(2)}" stroke="var(--gemma-border)" stroke-width="0.7" />
            <line x1="${cx + 2}" y1="88" x2="${cx + 28}" y2="88" stroke="var(--gemma-border-strong)" stroke-width="0.6" />
            <rect x="${cx + 7}" y="${barY}" width="16" height="${bh}" rx="1.5" fill="${fillCol}" />
            <text x="${cx + 15}" y="110" text-anchor="middle" font-size="6.4" font-family="monospace" font-weight="600" fill="var(--gemma-text-secondary)">${fmt}</text>
          `;
        })
        .join("");

      // Cosine angle arc for top neighbor
      const topNb = ranked[0];
      const clampedSim = Math.max(-1, Math.min(1, topNb.sim));
      const thetaRad = Math.acos(clampedSim);
      const thetaDeg = Math.round((thetaRad * 180) / Math.PI);
      const arcOriginX = 22;
      const arcOriginY = 30;
      const vecLen = 22;
      const bx = arcOriginX + Math.cos(-thetaRad) * vecLen;
      const by = arcOriginY + Math.sin(-thetaRad) * vecLen;
      const arcR = 12;
      const arcEx = arcOriginX + Math.cos(-thetaRad) * arcR;
      const arcEy = arcOriginY + Math.sin(-thetaRad) * arcR;
      const scaleX = (s) => 116 + ((s + 1) / 2) * 176;

      if (!bodyEl.querySelector('[data-role="space-stage"]')) {
        bodyEl.innerHTML = `
          <div class="llm-toolbar" data-role="space-toolbar" style="justify-content:space-between;margin-bottom:var(--gemma-space-2);">
            <div class="llm-toolbar__group" role="group" aria-label="Select vector scenario">
              <span class="llm-toolbar__label">Scenario</span>
              <button type="button" class="llm-scenario-btn" data-analogy="none">Free Explore</button>
              <button type="button" class="llm-scenario-btn" data-analogy="royalty">king − man + woman ≈ queen</button>
              <button type="button" class="llm-scenario-btn" data-analogy="pets">kitten − cat + dog ≈ puppy</button>
              <button type="button" class="llm-scenario-btn" data-analogy="multimodal">🖼 Image + 🔊 Audio ≈ ▁cat</button>
            </div>
            <div data-role="space-toolbar-right" style="display:inline-flex;align-items:center;gap:0.35rem;"></div>
          </div>
          <div data-role="space-stage"></div>
        `;
      }

      bodyEl.querySelectorAll('[data-role="space-toolbar"] [data-analogy]').forEach((btn) => {
        const key = btn.dataset.analogy;
        const isAct = key === "none" ? !analogyMode : analogyMode === key;
        btn.classList.toggle("llm-scenario-btn--active", isAct);
        btn.setAttribute("aria-pressed", String(isAct));
      });
      const spaceRightEl = bodyEl.querySelector('[data-role="space-toolbar-right"]');
      if (spaceRightEl) {
        if (spaceRightEl.dataset.modeKey !== modeKey) {
          spaceRightEl.dataset.modeKey = modeKey;
          if (modeKey === "free") {
            spaceRightEl.innerHTML = `
              <button type="button" class="llm-action-btn" data-action="tour-space"></button>
              <button type="button" class="llm-action-btn" data-action="jump-ple">
                ⚡ PLE Layer Injection &rarr;
              </button>
            `;
          } else {
            spaceRightEl.innerHTML = `
              <div class="llm-toolbar__group">
                <div class="llm-seg">
                  ${ANALOGY_CONFIGS[analogyMode].steps
                    .map(
                      (st, i) => `
                      <button type="button" class="llm-seg__btn"
                        data-analogy-step="${i}" style="font-size:0.7rem;padding:0.16rem 0.46rem;">${escapeHtml(st)}</button>`
                    )
                    .join("")}
                </div>
                <button type="button" class="llm-action-btn" data-action="replay-analogy" style="padding:0.2rem 0.55rem;"></button>
              </div>
            `;
          }
        }
        if (modeKey === "free") {
          const tourBtn = spaceRightEl.querySelector('[data-action="tour-space"]');
          if (tourBtn) {
            tourBtn.classList.toggle("llm-action-btn--running", Boolean(spaceTourTimer));
            tourBtn.classList.toggle("llm-btn--primary", !spaceTourTimer);
            const nextTxt = spaceTourTimer ? "⏸ Stop" : "▶ Auto-Tour";
            if (tourBtn.textContent !== nextTxt) tourBtn.textContent = nextTxt;
          }
        } else {
          spaceRightEl.querySelectorAll("[data-analogy-step]").forEach((b) => {
            const isAct = Number(b.dataset.analogyStep) === analogyStep;
            b.classList.toggle("llm-seg__btn--active", isAct);
            b.setAttribute("aria-pressed", String(isAct));
          });
          const replayBtn = spaceRightEl.querySelector('[data-action="replay-analogy"]');
          if (replayBtn) {
            replayBtn.classList.toggle("llm-action-btn--running", Boolean(analogyTimer));
            replayBtn.classList.toggle("llm-btn--primary", !analogyTimer);
            const nextTxt = analogyTimer ? "⏸ Stop" : "↻ Replay";
            if (replayBtn.textContent !== nextTxt) replayBtn.textContent = nextTxt;
          }
        }
      }

      const spaceStageEl = bodyEl.querySelector('[data-role="space-stage"]');
      spaceStageEl.innerHTML = `
        <div class="llm-split">
          <svg class="llm-svg-stage" viewBox="0 0 420 264" data-role="canvas" aria-label="2D semantic projection of Gemma 4 embedding space">
            <!-- Semantic Cluster Regions -->
            <rect x="8" y="8" width="194" height="126" rx="8" fill="var(--gemma-cat-turn-bg)" fill-opacity="0.34" stroke="var(--gemma-cat-turn-border)" stroke-opacity="0.6" />
            <rect x="210" y="8" width="202" height="126" rx="8" fill="var(--gemma-cat-think-bg)" fill-opacity="0.3" stroke="var(--gemma-cat-think-border)" stroke-opacity="0.6" />
            <rect x="8" y="142" width="194" height="114" rx="8" fill="var(--gemma-cat-tool-bg)" fill-opacity="0.28" stroke="var(--gemma-cat-tool-border)" stroke-opacity="0.6" />
            <rect x="210" y="142" width="202" height="114" rx="8" fill="var(--gemma-cat-multi-bg)" fill-opacity="0.3" stroke="var(--gemma-cat-multi-border)" stroke-opacity="0.6" />
            ${gridLines.join("")}
            <text x="16" y="20" fill="var(--gemma-cat-turn-fg)" font-size="8" font-family="monospace" font-weight="700">ANIMALS &amp; MULTIMODAL</text>
            <text x="218" y="20" fill="var(--gemma-cat-think-fg)" font-size="8" font-family="monospace" font-weight="700">ROYALTY &amp; PEOPLE</text>
            <text x="16" y="249" fill="var(--gemma-cat-tool-fg)" font-size="8" font-family="monospace" font-weight="700" opacity="${analogyMode ? "0.28" : "0.9"}">ACTIONS &amp; CODE</text>
            <text x="404" y="249" text-anchor="end" fill="var(--gemma-cat-multi-fg)" font-size="8" font-family="monospace" font-weight="700" opacity="${analogyMode ? "0.28" : "0.9"}">TECH &amp; AI</text>
            ${raysSvg}
            ${sonarSvg}
            ${overlaySvg}
            ${EMBED_POINTS.map((pt, idx) => {
              const isSel = idx === activeIdx;
              const activeCfg = analogyMode ? ANALOGY_CONFIGS[analogyMode] : null;
              const isInAnalogy = !activeCfg || activeCfg.activeSet.includes(idx);
              const ptOpacity = activeCfg && !isInAnalogy ? "0.22" : "1";
              const disp = pt.shortLabel || pt.label;
              const marker = pt.isSoftToken
                ? `<rect x="${pt.x - 6}" y="${pt.y - 6}" width="12" height="12" rx="3"
                    fill="${isSel ? "var(--gemma-accent-fill)" : "var(--gemma-cat-multi-bg)"}"
                    stroke="var(--gemma-accent)" stroke-width="${isSel ? 2.2 : 1.4}" stroke-dasharray="${isSel ? "" : "2.5 1.5"}" />`
                : `<circle cx="${pt.x}" cy="${pt.y}" r="${isSel ? 6.8 : 4.8}"
                    fill="${isSel ? "var(--gemma-accent-fill)" : "var(--gemma-surface)"}"
                    stroke="var(--gemma-accent)" stroke-width="${isSel ? 2.4 : 1.5}" />`;
              return `
                <g data-pt="${idx}" opacity="${ptOpacity}" style="cursor:pointer;transition:opacity 0.2s ease;">
                  ${marker}
                  <text x="${pt.x + 8.5}" y="${pt.y + 3.2}"
                    fill="${isSel ? "var(--md-default-fg-color)" : "var(--gemma-text-secondary)"}"
                    font-size="${isSel ? "10.5" : "9.5"}"
                    font-weight="${isSel ? "700" : "600"}"
                    font-family="var(--md-code-font-family, monospace)">${escapeHtml(disp)}</text>
                </g>
              `;
            }).join("")}
          </svg>

          <div style="display:flex;flex-direction:column;gap:var(--gemma-space-2);">
            <!-- Card 1: O(1) Embedding Lookup + Verified E2B BF16 [:8] Heatmap -->
            <svg class="llm-svg-stage" viewBox="0 0 304 118" aria-label="O(1) embedding table row lookup and BF16 vector slice">
              <text x="10" y="14" font-size="8" font-weight="700" font-family="monospace" fill="var(--gemma-accent)">${target.isSoftToken ? "SOFT-TOKEN PROJECTION" : "O(1) TABLE LOOKUP · 262K × 1,536"}</text>
              <text x="294" y="14" text-anchor="end" font-size="7.5" font-weight="700" font-family="monospace" fill="var(--gemma-text-muted)">ID #${target.id}</text>

              <!-- Token ID box -->
              <rect x="8" y="22" width="78" height="34" rx="5" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-accent)" stroke-width="1.2" />
              <text x="47" y="36" text-anchor="middle" font-size="8.8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-turn-fg)">${escapeHtml(target.shortLabel || target.label)}</text>
              <text x="47" y="48" text-anchor="middle" font-size="7" font-family="monospace" fill="var(--gemma-text-secondary)">#${target.id}</text>

              <line x1="86" y1="39" x2="110" y2="39" stroke="var(--gemma-accent)" stroke-width="1.4" stroke-dasharray="3 2" data-flow-wire="22" />
              <circle r="2.5" fill="var(--gemma-accent-fill)" data-anim-pts="86,39;110,39" data-dur="620" data-phase="0" />

              ${lookupMiddleSvg}

              <line x1="206" y1="39" x2="226" y2="39" stroke="var(--gemma-accent)" stroke-width="1.4" stroke-dasharray="3 2" data-flow-wire="22" />
              <circle r="2.5" fill="var(--gemma-accent-fill)" data-anim-pts="206,39;226,39" data-dur="620" data-phase="0.45" />

              <!-- Output h0 + clickable PLE jump -->
              <g data-action="jump-ple" style="cursor:pointer">
                <rect x="226" y="21" width="70" height="36" rx="5" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-status-warning)" stroke-width="1.1" />
                <text x="261" y="35" text-anchor="middle" font-size="8" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">h₀ 1,536d</text>
                <text x="261" y="48" text-anchor="middle" font-size="6.8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">+35×256d PLE ↗</text>
              </g>

              <!-- Verified BF16 [:8] Heatmap -->
              <text x="10" y="69" font-size="7.2" font-weight="700" font-family="monospace" fill="var(--gemma-text-secondary)">Gemma 4 E2B embed_tokens[${target.id}][:8] (BF16)</text>
              <text x="296" y="69" text-anchor="end" font-size="7" font-family="monospace" fill="var(--gemma-text-muted)">… 1,536</text>
              ${heatmapCellsSvg}
              <text x="285" y="92" text-anchor="middle" font-size="8.5" font-weight="700" font-family="monospace" fill="var(--gemma-text-muted)">⋯</text>
            </svg>

            <!-- Card 2: Cosine Angle Gauge + Interactive Nearest Neighbors -->
            <div class="llm-panel" style="margin-bottom:0;padding:0.5rem 0.65rem;">
              <svg viewBox="0 0 304 38" style="width:100%;height:auto;display:block;margin-bottom:0.3rem;" aria-label="Cosine angle and similarity scale">
                <!-- Angle quadrant -->
                <line x1="${arcOriginX}" y1="${arcOriginY}" x2="${arcOriginX + vecLen}" y2="${arcOriginY}" stroke="var(--gemma-accent)" stroke-width="1.8" stroke-linecap="round" />
                <line x1="${arcOriginX}" y1="${arcOriginY}" x2="${bx.toFixed(1)}" y2="${by.toFixed(1)}" stroke="var(--gemma-status-success)" stroke-width="1.8" stroke-linecap="round" />
                <path d="M ${arcOriginX + arcR} ${arcOriginY} A ${arcR} ${arcR} 0 0 0 ${arcEx.toFixed(1)} ${arcEy.toFixed(1)}" fill="none" stroke="var(--gemma-status-warning)" stroke-width="1.2" />
                <text x="52" y="16" font-size="7.8" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">cos(θ) = ${topNb.sim.toFixed(2)}</text>
                <text x="52" y="28" font-size="7.2" font-family="monospace" fill="var(--gemma-text-muted)">θ ≈ ${thetaDeg}° (${escapeHtml(topNb.shortLabel || topNb.label)})</text>

                <!-- -1..+1 cosine axis -->
                <line x1="116" y1="22" x2="292" y2="22" stroke="var(--gemma-border-strong)" stroke-width="2" stroke-linecap="round" />
                <line x1="204" y1="17" x2="204" y2="27" stroke="var(--gemma-border-strong)" stroke-width="1" />
                <text x="116" y="13" font-size="6.8" font-family="monospace" fill="var(--gemma-text-muted)">−1</text>
                <text x="204" y="13" text-anchor="middle" font-size="6.8" font-family="monospace" fill="var(--gemma-text-muted)">0</text>
                <text x="292" y="13" text-anchor="end" font-size="6.8" font-family="monospace" fill="var(--gemma-accent)" font-weight="700">+1</text>
                ${ranked
                  .slice()
                  .reverse()
                  .map(
                    (r, i) => `
                    <circle cx="${scaleX(r.sim).toFixed(1)}" cy="22" r="${i === ranked.length - 1 ? "4" : "2.6"}"
                      fill="${i === ranked.length - 1 ? "var(--gemma-status-success)" : "var(--gemma-accent-fill)"}"
                      fill-opacity="${i === ranked.length - 1 ? "1" : "0.5"}" />`
                  )
                  .join("")}
              </svg>
              ${ranked
                .map((r) => {
                  const pct = Math.max(5, Math.round(((r.sim + 1) / 2) * 100));
                  return `
                    <div data-pt="${r.idx}" style="display:grid;grid-template-columns:5.2rem 3.4rem 1fr 2.6rem;gap:0.35rem;align-items:center;font-size:0.73rem;padding:0.14rem 0.2rem;border-radius:var(--gemma-radius-sm);cursor:pointer;">
                      <code style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(r.shortLabel || r.label)}</code>
                      <span style="font-family:monospace;font-size:0.65rem;color:var(--gemma-text-muted);">#${r.id}</span>
                      <div class="llm-bar-track"><div class="llm-bar-fill" style="width:${pct}%"></div></div>
                      <strong style="text-align:right;font-family:monospace;">${r.sim.toFixed(2)}</strong>
                    </div>`;
                })
                .join("")}
            </div>
          </div>
        </div>
      `;
    };

    const renderPleTab = () => {
      const m = PLE_MODELS[pleModIdx];
      if (selectedPleLayer > m.layers) selectedPleLayer = m.layers;
      if (selectedPleLayer < 1) selectedPleLayer = 1;
      const target = EMBED_POINTS[activeIdx];
      const sliceHot = pleSubphase === 0;
      const gateHot = pleSubphase === 1;

      const sampleIdxs = PLE_SAMPLE_IDXS.includes(activeIdx) ? PLE_SAMPLE_IDXS : [...PLE_SAMPLE_IDXS, activeIdx];

      // --- Top Wide SVG (680 x 140): Dual O(1) Lookup + Layer-by-Layer PLE Bus ---
      const trackX = 172;
      const trackW = 494;
      const step = trackW / m.layers;
      const colW = Math.max(4.8, Math.min(11.8, step - 2.1));
      const cx = (lNum) => trackX + (lNum - 0.5) * step;

      const earlyEnd = Math.round(m.layers * 0.29);
      const midEnd = Math.round(m.layers * 0.68);
      const activeBand = selectedPleLayer <= earlyEnd ? 0 : selectedPleLayer <= midEnd ? 1 : 2;
      const bandLabels = ["Syntax / POS", "Semantics / Entity", "Phrasing / Next-Tok"];

      const layerColsSvg = Array.from({ length: m.layers }, (_, i) => {
        const lNum = i + 1;
        const x = cx(lNum) - colW / 2;
        const isAct = lNum === selectedPleLayer;
        const isFutureInCascade = Boolean(pleCascadeTimer && lNum > selectedPleLayer);
        const lift = isAct && pleCascadeTimer ? -2.5 : 0;
        const showTick =
          lNum === 1 ||
          lNum === m.layers ||
          lNum === earlyEnd ||
          lNum === midEnd ||
          isAct;

        if (m.hasPle) {
          return `
            <g data-ple-layer="${lNum}" style="cursor:pointer" ${isFutureInCascade ? 'opacity="0.42"' : ""} ${lift ? `transform="translate(0, ${lift})"` : ""}>
              <rect x="${cx(lNum) - step / 2}" y="38" width="${step}" height="92" fill="transparent" />
              <line x1="${cx(lNum)}" y1="40" x2="${cx(lNum)}" y2="54"
                stroke="${isAct ? "var(--gemma-status-warning)" : "var(--gemma-cat-think-border)"}"
                stroke-width="${isAct ? "1.8" : "0.9"}" stroke-opacity="${isAct ? "1" : "0.55"}" />
              ${
                isAct
                  ? `<rect x="${x - 2}" y="51.5" width="${colW + 4}" height="59" rx="3.5" fill="none"
                      stroke="${gateHot ? "var(--gemma-status-success)" : "var(--gemma-accent)"}" stroke-width="1.8" />`
                  : ""
              }
              <!-- 256-d PLE slice cap -->
              <rect x="${x}" y="54" width="${colW}" height="11" rx="2"
                fill="${isAct ? "var(--gemma-status-warning)" : "var(--gemma-cat-think-bg)"}"
                stroke="${isAct ? "var(--gemma-status-warning)" : "var(--gemma-cat-think-border)"}" />
              <!-- Decoder Attn+FFN core -->
              <rect x="${x}" y="68" width="${colW}" height="28" rx="2"
                fill="${isAct ? "var(--gemma-accent-fill)" : "var(--gemma-surface-sunken)"}"
                stroke="${isAct ? "var(--gemma-accent)" : "var(--gemma-border-strong)"}" />
              <!-- Gated residual merge node -->
              <circle cx="${cx(lNum)}" cy="105" r="${isAct ? "3.4" : "2.1"}"
                fill="${isAct ? "var(--gemma-status-success)" : "var(--gemma-cat-tool-fg)"}" />
              ${
                showTick
                  ? `<text x="${cx(lNum)}" y="125" text-anchor="middle" font-size="${isAct ? "8.2" : "7"}" font-weight="${isAct ? "700" : "500"}"
                      font-family="monospace" fill="${isAct ? "var(--gemma-accent)" : "var(--gemma-text-muted)"}">L${lNum}</text>`
                  : ""
              }
            </g>
          `;
        }

        return `
          <g data-ple-layer="${lNum}" style="cursor:pointer" ${isFutureInCascade ? 'opacity="0.42"' : ""} ${lift ? `transform="translate(0, ${lift})"` : ""}>
            <rect x="${cx(lNum) - step / 2}" y="50" width="${step}" height="80" fill="transparent" />
            ${
              isAct
                ? `<rect x="${x - 2}" y="53" width="${colW + 4}" height="56" rx="3.5" fill="none" stroke="var(--gemma-accent)" stroke-width="1.8" />`
                : ""
            }
            <rect x="${x}" y="56" width="${colW}" height="40" rx="2"
              fill="${isAct ? "var(--gemma-accent-fill)" : "var(--gemma-surface-sunken)"}"
              stroke="${isAct ? "var(--gemma-accent)" : "var(--gemma-border-strong)"}" />
            <circle cx="${cx(lNum)}" cy="105" r="${isAct ? "3.2" : "2"}" fill="var(--gemma-accent-fill)" />
            ${
              showTick
                ? `<text x="${cx(lNum)}" y="125" text-anchor="middle" font-size="${isAct ? "8.2" : "7"}" font-weight="${isAct ? "700" : "500"}"
                    font-family="monospace" fill="${isAct ? "var(--gemma-accent)" : "var(--gemma-text-muted)"}">L${lNum}</text>`
                : ""
            }
          </g>
        `;
      }).join("");

      const pleBusSvg = m.hasPle
        ? `
          <!-- Specialization Zone Labels -->
          <text x="${(cx(1) + cx(earlyEnd)) / 2}" y="30" text-anchor="middle" font-size="7.2" font-family="monospace" font-weight="${activeBand === 0 ? "700" : "500"}" fill="${activeBand === 0 ? "var(--gemma-cat-think-fg)" : "var(--gemma-text-muted)"}">L1–L${earlyEnd} Syntax</text>
          <text x="${(cx(earlyEnd + 1) + cx(midEnd)) / 2}" y="30" text-anchor="middle" font-size="7.2" font-family="monospace" font-weight="${activeBand === 1 ? "700" : "500"}" fill="${activeBand === 1 ? "var(--gemma-cat-think-fg)" : "var(--gemma-text-muted)"}">L${earlyEnd + 1}–L${midEnd} Semantics</text>
          <text x="${(cx(midEnd + 1) + cx(m.layers)) / 2}" y="30" text-anchor="middle" font-size="7.2" font-family="monospace" font-weight="${activeBand === 2 ? "700" : "500"}" fill="${activeBand === 2 ? "var(--gemma-cat-think-fg)" : "var(--gemma-text-muted)"}">L${midEnd + 1}–L${m.layers} Phrasing</text>

          <!-- Flash PLE Distribution Rail -->
          <line x1="156" y1="40" x2="${cx(m.layers)}" y2="40" stroke="var(--gemma-status-warning)" stroke-width="1.8" stroke-dasharray="5 3" data-flow-wire="20" />
          <circle r="3.4" fill="var(--gemma-status-warning)" data-anim-pts="156,40;${cx(selectedPleLayer)},40;${cx(selectedPleLayer)},58" data-dur="780" data-phase="0" />
          <circle r="2.7" fill="var(--gemma-accent-fill)" data-anim-pts="156,40;${cx(selectedPleLayer)},40;${cx(selectedPleLayer)},58" data-dur="780" data-phase="0.5" />
        `
        : `
          <line x1="156" y1="38" x2="${cx(m.layers)}" y2="38" stroke="var(--gemma-border-strong)" stroke-width="1" stroke-dasharray="3 3" opacity="0.45" />
          <rect x="282" y="29" width="276" height="16" rx="8" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
          <text x="420" y="39.8" text-anchor="middle" font-size="7.4" font-family="monospace" font-weight="600" fill="var(--gemma-text-muted)">0B Flash PLE · Wide ${m.dModelFmt}-d Layer-0 stream in (V)RAM</text>
        `;

      // --- Bottom-Left SVG (334 x 168): Layer PLE Gated Injection Circuit ---
      const miniSparkBars = Array.from({ length: 10 }, (_, bi) => {
        const val = m.hasPle ? computePleChannelVal(activeIdx, selectedPleLayer, m.layers, bi * 2 + 2, Boolean(target.isSoftToken)) : 0;
        const bh = Math.max(2, Math.round(Math.abs(val) * 12));
        const sx = 86 + bi * 4.4;
        const sy = val >= 0 ? 139 - bh : 139;
        return `<rect x="${sx.toFixed(1)}" y="${sy}" width="3" height="${bh}" rx="1" fill="${val >= 0 ? "var(--gemma-status-warning)" : "var(--gemma-accent-fill)"}" />`;
      }).join("");

      const pleCircuitParticles = m.hasPle
        ? `
          <circle r="2.7" fill="var(--gemma-accent-fill)" data-anim-pts="50,44;285,44" data-dur="900" data-phase="0.1" />
          <circle r="2.7" fill="var(--gemma-status-warning)" data-anim-pts="56,44;56,80;178,80;192,80;192,99" data-dur="880" data-phase="0" />
          <circle r="2.8" fill="var(--gemma-status-warning)" data-anim-pts="136,134;192,134;192,117" data-dur="760" data-phase="0.2" />
          <circle r="2.9" fill="var(--gemma-status-success)" data-anim-pts="201,108;293,108;293,52" data-dur="820" data-phase="0.55" />
        `
        : `
          <circle r="3.1" fill="var(--gemma-accent-fill)" data-anim-pts="50,44;304,44" data-dur="680" data-phase="0" />
          <circle r="2.6" fill="var(--gemma-accent-fill)" data-anim-pts="50,44;304,44" data-dur="680" data-phase="0.5" />
        `;

      const circuitLeftSvg = `
        <svg class="llm-svg-stage" viewBox="0 0 334 168" style="${gateHot && m.hasPle ? "border-color:var(--gemma-status-success);box-shadow:var(--gemma-shadow-sm);" : sliceHot && m.hasPle ? "border-color:var(--gemma-status-warning);box-shadow:var(--gemma-shadow-sm);" : ""}" aria-label="Per-Layer Embedding gated injection circuit">
          <text x="10" y="16" font-size="8.8" font-weight="700" font-family="monospace" fill="${gateHot && m.hasPle ? "var(--gemma-cat-tool-fg)" : "var(--md-default-fg-color)"}">L${selectedPleLayer} · ${m.hasPle ? "PLE Gate Circuit" : "Bypassed (No PLE)"}</text>
          <text x="324" y="16" text-anchor="end" font-size="7.6" font-weight="700" font-family="monospace" fill="${m.hasPle ? "var(--gemma-cat-think-fg)" : "var(--gemma-text-muted)"}">${m.hasPle ? `${m.dModelFmt} → 256 → ${m.dModelFmt}` : `h_l (${m.dModelFmt}-d) direct`}</text>
          <line x1="8" y1="23" x2="326" y2="23" stroke="var(--gemma-border)" />

          <!-- Main Residual Stream h_l -> h'_l -->
          <rect x="10" y="33" width="40" height="22" rx="4" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-accent)" />
          <text x="30" y="47" text-anchor="middle" font-size="8.2" font-weight="700" font-family="monospace" fill="var(--gemma-cat-turn-fg)">h_l</text>

          <line x1="50" y1="44" x2="304" y2="44" stroke="var(--gemma-accent)" stroke-width="${m.hasPle ? "1.6" : "3"}" />

          <g opacity="${m.hasPle ? "1" : "0.28"}">
            <!-- Gate branch from h_l -->
            <path d="M 56 44 L 56 80 L 192 80 L 192 99.5" fill="none" stroke="var(--gemma-cat-think-border)" stroke-width="1.5" ${m.hasPle ? "" : 'stroke-dasharray="3 2"'} />
            <rect x="64" y="68" width="72" height="24" rx="4" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="100" y="79" text-anchor="middle" font-size="7.4" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">input_gate</text>
            <text x="100" y="88" text-anchor="middle" font-size="6.6" font-family="monospace" fill="var(--gemma-text-muted)">${m.hasPle ? `${m.dModelFmt}→256` : "✕ disabled"}</text>

            <rect x="142" y="68" width="36" height="24" rx="4" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-cat-think-border)" />
            <text x="160" y="83" text-anchor="middle" font-size="7.6" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">GELU</text>

            <!-- Flash Slice Input ple[L] -->
            <path d="M 136 134 L 192 134 L 192 116.5" fill="none" stroke="var(--gemma-status-warning)" stroke-width="1.6" ${m.hasPle ? "" : 'stroke-dasharray="3 2"'} />
            <rect x="10" y="114" width="126" height="40" rx="4" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-status-warning)" stroke-width="1.3" />
            <text x="16" y="127" font-size="7.4" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">ple[L${selectedPleLayer}]</text>
            <text x="16" y="138" font-size="6.8" font-family="monospace" fill="var(--gemma-text-secondary)">${target.isSoftToken ? "model_proj" : "Flash 256-d"}</text>
            <text x="16" y="148" font-size="6.5" font-family="monospace" fill="var(--gemma-text-muted)">${escapeHtml(target.shortLabel || target.label)} #${target.id}</text>
            ${miniSparkBars}

            <!-- Element-wise Gate Product -->
            <circle cx="192" cy="108" r="8.5" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-status-success)" stroke-width="1.4" />
            <text x="192" y="111.5" text-anchor="middle" font-size="9.5" font-weight="700" fill="var(--gemma-cat-tool-fg)">⊙</text>

            <!-- Up-projection + Post-PLE RMSNorm -> Residual Add -->
            <path d="M 200.5 108 L 293 108 L 293 52" fill="none" stroke="var(--gemma-status-success)" stroke-width="1.6" ${m.hasPle ? "" : 'stroke-dasharray="3 2"'} />
            <rect x="206" y="95" width="56" height="26" rx="4" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-cat-tool-border)" />
            <text x="234" y="106.5" text-anchor="middle" font-size="7.3" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">ple_proj</text>
            <text x="234" y="116" text-anchor="middle" font-size="6.5" font-family="monospace" fill="var(--gemma-text-secondary)">${m.hasPle ? `256→${m.dModelFmt}` : "—"}</text>

            <rect x="268" y="95" width="50" height="26" rx="4" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="293" y="106.5" text-anchor="middle" font-size="7" font-weight="700" font-family="monospace" fill="var(--gemma-text-secondary)">RMSNorm</text>
            <text x="293" y="116" text-anchor="middle" font-size="6.3" font-family="monospace" fill="var(--gemma-text-muted)">post_ple</text>

            <circle cx="293" cy="44" r="8" fill="var(--gemma-surface)" stroke="var(--gemma-status-success)" stroke-width="1.3" />
            <text x="293" y="47.2" text-anchor="middle" font-size="9.5" font-weight="700" fill="var(--gemma-cat-tool-fg)">⊕</text>
          </g>

          <rect x="304" y="33" width="22" height="22" rx="4" fill="${gateHot && m.hasPle ? "var(--gemma-status-success)" : "var(--gemma-accent-fill)"}" />
          <text x="315" y="47" text-anchor="middle" font-size="8" font-weight="700" font-family="monospace" fill="var(--gemma-white)">h'</text>

          ${pleCircuitParticles}
        </svg>
      `;

      // --- Bottom-Right SVG (334 x 168): (V)RAM vs. Flash Split + 24-Bar Gated 256-d Spectrum ---
      const compareIdxs = pleModIdx <= 2 ? [0, 1, 2] : [0, 1, pleModIdx];
      const maxBarCap = Math.max(...compareIdxs.map((idx) => PLE_MODELS[idx].totalB));
      const memRowsSvg = compareIdxs
        .map((cIdx, rowIdx) => {
          const cm = PLE_MODELS[cIdx];
          const ry = 27 + rowIdx * 20;
          const isCur = cIdx === pleModIdx;
          const barMaxW = 166;
          const vramW = Math.max(8, Math.round((cm.activeB / maxBarCap) * barMaxW));
          const flashW = cm.sparseB > 0 ? Math.max(8, Math.round((cm.sparseB / maxBarCap) * barMaxW)) : 0;
          return `
            <g data-ple-mod="${cIdx}" style="cursor:pointer">
              <rect x="6" y="${ry - 3}" width="322" height="17" rx="3.5" fill="${isCur ? "var(--gemma-cat-turn-bg)" : "transparent"}" fill-opacity="0.45" />
              <text x="10" y="${ry + 8.5}" font-size="7.8" font-weight="${isCur ? "700" : "600"}" font-family="monospace" fill="${isCur ? "var(--gemma-accent)" : "var(--gemma-text-secondary)"}">${cm.shortTag}</text>
              <rect x="40" y="${ry}" width="${vramW}" height="11" rx="2" fill="var(--gemma-accent-fill)" />
              ${
                flashW > 0
                  ? `<rect x="${40 + vramW + 1.5}" y="${ry}" width="${flashW}" height="11" rx="2" fill="var(--gemma-status-warning)" fill-opacity="0.85" />`
                  : ""
              }
              <text x="324" y="${ry + 8.5}" text-anchor="end" font-size="7.1" font-family="monospace" font-weight="${isCur ? "700" : "500"}" fill="${isCur ? "var(--md-default-fg-color)" : "var(--gemma-text-muted)"}">${cm.activeB.toFixed(1)}B + ${cm.sparseB.toFixed(1)}B Flash</text>
            </g>
          `;
        })
        .join("");

      const spectrumBarsSvg = Array.from({ length: 24 }, (_, k) => {
        const bx = 16 + k * 12.6;
        if (!m.hasPle) {
          return `<rect x="${bx.toFixed(1)}" y="136" width="8.5" height="2" rx="1" fill="var(--gemma-border-strong)" opacity="0.5" />`;
        }
        const val = computePleChannelVal(activeIdx, selectedPleLayer, m.layers, k, Boolean(target.isSoftToken));
        const bh = Math.max(2, Math.round(Math.abs(val) * 16));
        const by = val >= 0 ? 137 - bh : 137;
        const bandOfBar = k < 8 ? 0 : k < 16 ? 1 : 2;
        const isHotBand = bandOfBar === activeBand;
        const col = val >= 0 ? (isHotBand ? "var(--gemma-status-success)" : "var(--gemma-accent-fill)") : "var(--gemma-status-warning)";
        return `<rect x="${bx.toFixed(1)}" y="${by}" width="8.5" height="${bh}" rx="1.4" fill="${col}" fill-opacity="${isHotBand ? "0.95" : "0.45"}" />`;
      }).join("");

      const spectrumRightSvg = `
        <svg class="llm-svg-stage" viewBox="0 0 334 168" aria-label="(V)RAM vs Flash memory split and 256-d gated PLE spectrum">
          <text x="10" y="16" font-size="8.2" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">(V)RAM COMPUTE vs. FLASH LOOKUP</text>
          <g transform="translate(202, 7)" font-size="6.8" font-family="monospace">
            <rect x="0" y="1" width="7" height="7" rx="1.5" fill="var(--gemma-accent-fill)" />
            <text x="9" y="7" fill="var(--gemma-text-secondary)">(V)RAM</text>
            <rect x="46" y="1" width="7" height="7" rx="1.5" fill="var(--gemma-status-warning)" />
            <text x="55" y="7" fill="var(--gemma-text-secondary)">Flash PLE+Emb</text>
          </g>
          <line x1="8" y1="22" x2="326" y2="22" stroke="var(--gemma-border)" />
          ${memRowsSvg}

          <line x1="8" y1="92" x2="326" y2="92" stroke="var(--gemma-border)" />
          <text x="10" y="105" font-size="7.8" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">L${selectedPleLayer} · GELU ⊙ ple (${escapeHtml(target.shortLabel || target.label)})</text>
          <text x="324" y="105" text-anchor="end" font-size="7.4" font-weight="700" font-family="monospace" fill="${m.hasPle ? "var(--gemma-cat-tool-fg)" : "var(--gemma-text-muted)"}">${m.hasPle ? bandLabels[activeBand] : "0 PLE dims"}</text>

          <!-- 3 Subspace Band Highlights -->
          <rect x="12" y="112" width="101" height="42" rx="4" fill="${m.hasPle && activeBand === 0 ? "var(--gemma-cat-tool-bg)" : "var(--gemma-surface-sunken)"}" fill-opacity="${m.hasPle && activeBand === 0 ? "0.55" : "0.4"}" stroke="${m.hasPle && activeBand === 0 ? "var(--gemma-status-success)" : "var(--gemma-border)"}" />
          <rect x="116" y="112" width="101" height="42" rx="4" fill="${m.hasPle && activeBand === 1 ? "var(--gemma-cat-tool-bg)" : "var(--gemma-surface-sunken)"}" fill-opacity="${m.hasPle && activeBand === 1 ? "0.55" : "0.4"}" stroke="${m.hasPle && activeBand === 1 ? "var(--gemma-status-success)" : "var(--gemma-border)"}" />
          <rect x="220" y="112" width="102" height="42" rx="4" fill="${m.hasPle && activeBand === 2 ? "var(--gemma-cat-tool-bg)" : "var(--gemma-surface-sunken)"}" fill-opacity="${m.hasPle && activeBand === 2 ? "0.55" : "0.4"}" stroke="${m.hasPle && activeBand === 2 ? "var(--gemma-status-success)" : "var(--gemma-border)"}" />
          <line x1="14" y1="137" x2="320" y2="137" stroke="var(--gemma-border-strong)" stroke-width="0.6" />
          ${spectrumBarsSvg}

          <text x="62" y="163" text-anchor="middle" font-size="6.8" font-family="monospace" font-weight="${m.hasPle && activeBand === 0 ? "700" : "500"}" fill="${m.hasPle && activeBand === 0 ? "var(--gemma-cat-tool-fg)" : "var(--gemma-text-muted)"}">d₀..₈₅ Syntax</text>
          <text x="166" y="163" text-anchor="middle" font-size="6.8" font-family="monospace" font-weight="${m.hasPle && activeBand === 1 ? "700" : "500"}" fill="${m.hasPle && activeBand === 1 ? "var(--gemma-cat-tool-fg)" : "var(--gemma-text-muted)"}">d₈₆..₁₇₀ Semantics</text>
          <text x="271" y="163" text-anchor="middle" font-size="6.8" font-family="monospace" font-weight="${m.hasPle && activeBand === 2 ? "700" : "500"}" fill="${m.hasPle && activeBand === 2 ? "var(--gemma-cat-tool-fg)" : "var(--gemma-text-muted)"}">d₁₇₁..₂₅₅ Phrasing</text>
        </svg>
      `;

      if (!bodyEl.querySelector('[data-role="ple-stage"]')) {
        bodyEl.innerHTML = `
          <div class="llm-toolbar" data-role="ple-toolbar" style="justify-content:space-between;margin-bottom:var(--gemma-space-2);">
            <div class="llm-toolbar__group" role="group" aria-label="Select model and token for PLE">
              <span class="llm-toolbar__label">Model</span>
              ${PLE_MODELS.map(
                (mod, idx) =>
                  `<button type="button" class="llm-scenario-btn" data-ple-mod="${idx}">${escapeHtml(mod.pill)}</button>`
              ).join("")}
              <span class="llm-toolbar__sep" aria-hidden="true"></span>
              <span class="llm-toolbar__label">Token</span>
              <div class="llm-seg" data-role="ple-tokens"></div>
            </div>
            <button type="button" class="llm-action-btn" data-action="cascade-ple"></button>
          </div>
          <div data-role="ple-stage"></div>
        `;
      }

      bodyEl.querySelectorAll('[data-role="ple-toolbar"] [data-ple-mod]').forEach((btn) => {
        const isAct = Number(btn.dataset.pleMod) === pleModIdx;
        btn.classList.toggle("llm-scenario-btn--active", isAct);
        btn.setAttribute("aria-pressed", String(isAct));
      });

      const pleTokensEl = bodyEl.querySelector('[data-role="ple-tokens"]');
      const tokenKey = sampleIdxs.join(",");
      if (pleTokensEl) {
        if (pleTokensEl.dataset.tokenKey !== tokenKey) {
          pleTokensEl.dataset.tokenKey = tokenKey;
          pleTokensEl.innerHTML = sampleIdxs
            .map((pIdx) => {
              const p = EMBED_POINTS[pIdx];
              return `<button type="button" class="llm-seg__btn" data-pt="${pIdx}" style="font-family:monospace;font-size:0.71rem;padding:0.16rem 0.45rem;">${escapeHtml(p.shortLabel || p.label)}</button>`;
            })
            .join("");
        }
        pleTokensEl.querySelectorAll("[data-pt]").forEach((btn) => {
          const isAct = Number(btn.dataset.pt) === activeIdx;
          btn.classList.toggle("llm-seg__btn--active", isAct);
          btn.setAttribute("aria-pressed", String(isAct));
        });
      }

      const cascadeBtn = bodyEl.querySelector('[data-action="cascade-ple"]');
      if (cascadeBtn) {
        cascadeBtn.classList.toggle("llm-action-btn--running", Boolean(pleCascadeTimer));
        cascadeBtn.classList.toggle("llm-btn--primary", !pleCascadeTimer);
        const nextLabel = pleCascadeTimer
          ? "⏸ Stop"
          : `▶ ${m.hasPle ? "Inject PLE" : "Forward Pass"} (L1 → L${m.layers})`;
        if (cascadeBtn.textContent !== nextLabel) cascadeBtn.textContent = nextLabel;
      }

      const pleStageEl = bodyEl.querySelector('[data-role="ple-stage"]');
      pleStageEl.innerHTML = `
        <svg class="llm-svg-stage" viewBox="0 0 680 134" aria-label="${escapeHtml(m.fullName)} dual O(1) lookup and per-layer embedding bus">
          <text x="12" y="15" font-size="8.6" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">${escapeHtml(m.fullName)} · ${m.hasPle ? `O(1) Flash PLE (${m.layers}×256d) + ${m.dModelFmt}d Residual` : `Layer-0 Only (${m.dModelFmt}d Residual · No PLE)`}</text>
          <g transform="translate(422, 6)" font-size="7.4" font-family="monospace">
            <rect x="0" y="1" width="8.5" height="8.5" rx="2" fill="var(--gemma-status-warning)" />
            <text x="11" y="8" fill="var(--gemma-text-secondary)">Flash PLE (256d)</text>
            <rect x="96" y="1" width="8.5" height="8.5" rx="2" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="107" y="8" fill="var(--gemma-text-secondary)">Attn+FFN</text>
            <circle cx="166" cy="5.2" r="3.5" fill="var(--gemma-status-success)" />
            <text x="173" y="8" fill="var(--gemma-text-secondary)">Gate ⊕</text>
          </g>

          <!-- Top-left: Flash Sparse PLE Lookup Box -->
          <g opacity="${m.hasPle ? "1" : "0.35"}">
            <rect x="12" y="24" width="144" height="46" rx="5" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-status-warning)" stroke-width="1.2" ${m.hasPle ? "" : 'stroke-dasharray="3 2"'} />
            <text x="19" y="36" font-size="7" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">FLASH · SPARSE O(1) PLE</text>
            <text x="19" y="48" font-size="7.5" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">embed_tokens_per_layer</text>
            <text x="19" y="60" font-size="6.9" font-family="monospace" fill="var(--gemma-text-secondary)">${
              !m.hasPle
                ? "0B · Not present"
                : target.isSoftToken
                  ? `pad #0 + model_proj (${m.layers}×256d)`
                  : `262K×${m.layers}×256d · ${m.pleB}B`
            }</text>
          </g>

          <!-- Bottom-left: (V)RAM Layer-0 Lookup Box -->
          <rect x="12" y="78" width="144" height="46" rx="5" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-accent)" stroke-width="1.2" />
          <text x="19" y="90" font-size="7" font-weight="700" font-family="monospace" fill="var(--gemma-accent)">(V)RAM · LAYER-0 EMBED</text>
          <text x="19" y="102" font-size="7.5" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">${target.isSoftToken ? "multimodal_proj" : "embed_tokens"} (#${target.id})</text>
          <text x="19" y="114" font-size="6.9" font-family="monospace" fill="var(--gemma-text-secondary)">262K×${m.dModelFmt}d · ${m.embB.toFixed(2)}B</text>

          <!-- Main Residual Stream Bus -->
          <line x1="156" y1="105" x2="${cx(m.layers)}" y2="105" stroke="var(--gemma-accent)" stroke-width="${m.hasPle ? "1.8" : "3.4"}" />
          <circle r="2.8" fill="var(--gemma-accent-fill)" data-anim-pts="156,105;${cx(m.layers)},105" data-dur="1250" data-phase="0.25" />

          ${pleBusSvg}
          ${layerColsSvg}
        </svg>

        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(17.5rem,1fr));gap:var(--gemma-space-2);margin-top:var(--gemma-space-2);">
          ${circuitLeftSvg}
          ${spectrumRightSvg}
        </div>
        <p style="margin:var(--gemma-space-2) 0 0;font-size:0.74rem;line-height:1.45;color:var(--gemma-text-muted);"><strong style="color:var(--gemma-text-secondary);">Note:</strong> The strict split into <code>Syntax</code>, <code>Semantics</code>, and <code>Phrasing</code> subspaces on the bottom-right visualization is a didactic simplification. In reality, neural networks learn continuous, distributed representations.</p>
      `;
    };

    const render = () => {
      el.querySelectorAll("[data-emb-tab]").forEach((b) => {
        const isActive = b.dataset.embTab === tab;
        b.classList.toggle("llm-seg__btn--active", isActive);
        b.setAttribute("aria-pressed", String(isActive));
      });

      if (tab === "space") {
        renderSpaceTab();
      } else {
        renderPleTab();
      }

      cacheAnimNodes();
      stepParticles(typeof performance !== "undefined" ? performance.now() : Date.now());
      ensureRaf();
    };

    el.addEventListener("pointerdown", (e) => {
      if ((pleCascadeTimer || analogyTimer || spaceTourTimer) && e.target.closest("[data-ple-layer], [data-ple-mod], [data-pt], [data-analogy], [data-analogy-step]")) {
        stopTimers();
      }
    });

    el.addEventListener("click", (e) => {
      const tabBtn = e.target.closest("[data-emb-tab]");
      if (tabBtn) {
        stopTimers();
        tab = tabBtn.dataset.embTab;
        render();
        return;
      }

      if (e.target.closest('[data-action="jump-ple"]')) {
        stopTimers();
        tab = "ple";
        render();
        return;
      }

      if (e.target.closest('[data-action="tour-space"]')) {
        if (spaceTourTimer) {
          stopTimers();
          render();
          return;
        }
        stopTimers();
        analogyMode = null;
        let tourPos = Math.max(0, SPACE_TOUR_ORDER.indexOf(activeIdx));
        tourPos = (tourPos + 1) % SPACE_TOUR_ORDER.length;
        activeIdx = SPACE_TOUR_ORDER[tourPos];
        spaceTourTimer = setInterval(() => {
          if (!el.isConnected) {
            stopTimers();
            return;
          }
          tourPos = (tourPos + 1) % SPACE_TOUR_ORDER.length;
          activeIdx = SPACE_TOUR_ORDER[tourPos];
          render();
        }, 1150);
        render();
        return;
      }

      if (e.target.closest('[data-action="replay-analogy"]')) {
        if (analogyTimer) {
          stopTimers();
          render();
          return;
        }
        startAnalogyAnim();
        return;
      }

      if (e.target.closest('[data-action="cascade-ple"]')) {
        if (pleCascadeTimer) {
          stopTimers();
          render();
          return;
        }
        stopTimers();
        selectedPleLayer = 1;
        pleSubphase = 0;
        pleCascadeTimer = setInterval(() => {
          if (!el.isConnected) {
            stopTimers();
            return;
          }
          if (pleSubphase === 0 && PLE_MODELS[pleModIdx].hasPle) {
            pleSubphase = 1;
            render();
            return;
          }
          if (selectedPleLayer >= PLE_MODELS[pleModIdx].layers) {
            stopTimers();
            render();
            return;
          }
          selectedPleLayer += 1;
          pleSubphase = 0;
          render();
        }, 110);
        render();
        return;
      }

      const pleModBtn = e.target.closest("[data-ple-mod]");
      if (pleModBtn) {
        stopTimers();
        pleModIdx = Number(pleModBtn.dataset.pleMod);
        const layers = PLE_MODELS[pleModIdx].layers;
        if (selectedPleLayer > layers) selectedPleLayer = Math.ceil(layers / 2);
        render();
        return;
      }

      const pleLayerBtn = e.target.closest("[data-ple-layer]");
      if (pleLayerBtn) {
        stopTimers();
        selectedPleLayer = Number(pleLayerBtn.dataset.pleLayer);
        render();
        return;
      }

      const stepBtn = e.target.closest("[data-analogy-step]");
      if (stepBtn) {
        stopTimers();
        analogyStep = Number(stepBtn.dataset.analogyStep);
        if (analogyStep === 1) slideStartMs = nowMs();
        syncAnalogyTargetIdx();
        render();
        return;
      }

      const analogyBtn = e.target.closest("[data-analogy]");
      if (analogyBtn) {
        const chosen = analogyBtn.dataset.analogy;
        analogyMode = chosen === "none" || analogyMode === chosen ? null : chosen;
        if (!analogyMode) {
          stopTimers();
          render();
          return;
        }
        startAnalogyAnim();
        return;
      }

      const ptGroup = e.target.closest("[data-pt]");
      if (ptGroup) {
        stopTimers();
        activeIdx = Number(ptGroup.dataset.pt);
        if (tab === "space") analogyMode = null;
        render();
      }
    });

    const params = typeof window !== "undefined" && window.location ? new URLSearchParams(window.location.search) : null;
    const initAnalogy = params ? params.get("analogy") : null;
    const initStep = params && params.get("step") !== null ? Number(params.get("step")) : null;
    const initTab = params ? params.get("embTab") || params.get("ple") : null;
    if (initTab === "ple" || initTab === "1") {
      tab = "ple";
    }
    if (initAnalogy && ANALOGY_CONFIGS[initAnalogy]) {
      analogyMode = initAnalogy;
      analogyStep = Number.isInteger(initStep) && initStep >= 0 && initStep <= 2 ? initStep : 2;
      const cfg = ANALOGY_CONFIGS[analogyMode];
      activeIdx = analogyStep === 0 ? cfg.idxA : analogyStep === 1 ? cfg.idxC : cfg.idxD;
    }
    render();
  };

  /* =====================================================================
     3. SELF-ATTENTION & HYBRID LOCAL/GLOBAL LAYERS (Step 3)
     ===================================================================== */
  const ATTN_SENTENCES = [
    {
      label: '"polished" → trophy',
      words: ["Mia", "took", "the", "trophy", "from", "the", "suitcase", "and", "polished", "it", "again."],
      focusIdx: 9,
      weights: {
        3: [0.08, 0.24, 0.28, 0.40, 0, 0, 0, 0, 0, 0, 0],
        6: [0.05, 0.14, 0.03, 0.12, 0.18, 0.20, 0.28, 0, 0, 0, 0],
        8: [0.22, 0.06, 0.03, 0.46, 0.02, 0.02, 0.05, 0.04, 0.10, 0, 0],
        9: [0.03, 0.03, 0.02, 0.54, 0.02, 0.02, 0.05, 0.02, 0.20, 0.07, 0],
        10: [0.04, 0.08, 0.01, 0.18, 0.01, 0.01, 0.02, 0.02, 0.36, 0.21, 0.06],
      },
    },
    {
      label: '"zipped" → suitcase',
      words: ["Mia", "took", "the", "trophy", "from", "the", "suitcase", "and", "zipped", "it", "again."],
      focusIdx: 9,
      weights: {
        3: [0.08, 0.24, 0.28, 0.40, 0, 0, 0, 0, 0, 0, 0],
        6: [0.05, 0.14, 0.03, 0.12, 0.18, 0.20, 0.28, 0, 0, 0, 0],
        8: [0.20, 0.05, 0.02, 0.05, 0.03, 0.03, 0.48, 0.04, 0.10, 0, 0],
        9: [0.03, 0.03, 0.02, 0.04, 0.02, 0.02, 0.56, 0.02, 0.20, 0.06, 0],
        10: [0.04, 0.06, 0.01, 0.02, 0.01, 0.01, 0.20, 0.02, 0.36, 0.21, 0.06],
      },
    },
    {
      label: 'Late clue: "fragile."',
      words: ["The", "robot", "picked", "up", "the", "glass", "ball", "because", "it", "was", "fragile."],
      focusIdx: 10,
      weights: {
        1: [0.35, 0.65, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        6: [0.04, 0.10, 0.14, 0.04, 0.08, 0.36, 0.24, 0, 0, 0, 0],
        8: [0.03, 0.33, 0.04, 0.01, 0.03, 0.14, 0.34, 0.04, 0.04, 0, 0],
        9: [0.02, 0.24, 0.04, 0.01, 0.02, 0.10, 0.25, 0.06, 0.20, 0.06, 0],
        10: [0.01, 0.03, 0.02, 0.01, 0.02, 0.33, 0.35, 0.02, 0.15, 0.04, 0.02],
      },
    },
  ];

  const GEMMA_MODELS_ARCH = [
    {
      id: "e2b",
      name: "E2B · 35L",
      fullName: "Gemma 4 E2B",
      layers: 35,
      period: 5, // 4 local + 1 global
      windowShort: "512w",
      globalCtxShort: "128K",
      dModel: "1,536",
      dFf: "6,144",
      dFfWide: "12,288",
      doubleWideKvShared: true,
      qHeads: 8,
      localKvHeads: 1,
      globalKvHeads: 1,
      localGqa: "8:1",
      globalGqa: "8:1",
      kEqV: false,
      kvSharedLayers: 20,
    },
    {
      id: "e4b",
      name: "E4B · 42L",
      fullName: "Gemma 4 E4B",
      layers: 42,
      period: 6, // 5 local + 1 global
      windowShort: "512w",
      globalCtxShort: "128K",
      dModel: "2,560",
      dFf: "10,240",
      doubleWideKvShared: false,
      qHeads: 8,
      localKvHeads: 2,
      globalKvHeads: 2,
      localGqa: "4:1",
      globalGqa: "4:1",
      kEqV: false,
      kvSharedLayers: 18,
    },
    {
      id: "12b",
      name: "12B · 48L",
      fullName: "Gemma 4 12B",
      layers: 48,
      period: 6,
      windowShort: "1,024w",
      globalCtxShort: "256K",
      dModel: "3,840",
      dFf: "15,360",
      doubleWideKvShared: false,
      qHeads: 16,
      localKvHeads: 8,
      globalKvHeads: 1,
      localGqa: "2:1",
      globalGqa: "16:1",
      kEqV: true,
      kvSharedLayers: 0,
    },
    {
      id: "26b",
      name: "26B A4B · 30L",
      fullName: "Gemma 4 26B A4B MoE",
      layers: 30,
      period: 6,
      windowShort: "1,024w",
      globalCtxShort: "256K",
      dModel: "2,816",
      dFf: "704",
      doubleWideKvShared: false,
      qHeads: 16,
      localKvHeads: 8,
      globalKvHeads: 2,
      localGqa: "2:1",
      globalGqa: "8:1",
      kEqV: true,
      kvSharedLayers: 0,
    },
    {
      id: "31b",
      name: "31B · 60L",
      fullName: "Gemma 4 31B",
      layers: 60,
      period: 6,
      windowShort: "1,024w",
      globalCtxShort: "256K",
      dModel: "5,376",
      dFf: "21,504",
      doubleWideKvShared: false,
      qHeads: 32,
      localKvHeads: 16,
      globalKvHeads: 4,
      localGqa: "2:1",
      globalGqa: "8:1",
      kEqV: true,
      kvSharedLayers: 0,
    },
  ];

  const getDefaultLayerForModel = (m) => {
    if (m.kvSharedLayers > 0) {
      const firstShared = m.layers - m.kvSharedLayers + 1;
      const rem = firstShared % m.period;
      return rem === 0 ? firstShared : firstShared + (m.period - rem);
    }
    return m.period;
  };

  const mountAttention = (el) => {
    let tab = "context"; // "context" | "layers"
    let sentIdx = 0;
    let wordIdx = ATTN_SENTENCES[0].focusIdx;
    let modelIdx = 0;
    let selectedLayer = getDefaultLayerForModel(GEMMA_MODELS_ARCH[0]);
    let scanSubphase = -1; // -1 idle, 0 Sublayer A (Attn), 1 Sublayer B (FFN)
    let layerSubphase = -1; // -1 idle, 0 Sublayer A (Attn/KV), 1 Sublayer B (FFN/MoE)
    let scanTimer = null;
    let cascadeTimer = null;
    let rafId = 0;
    let bezParticles = [];
    let polyParticles = [];
    let flowWires = [];

    const panelEl = el.closest("[data-llm-panel]");
    const isVisible = () => el.isConnected && (!panelEl || !panelEl.hidden);

    el.innerHTML = `
      <div class="llm-widget__header">
        <p class="llm-widget__title">Causal Attention, KV Cache &amp; FFN / MoE</p>
        <div class="llm-toolbar__group" role="group" aria-label="Attention view mode">
          <span class="llm-toolbar__label">View</span>
          <div class="llm-seg">
            <button type="button" class="llm-seg__btn llm-seg__btn--active" data-tab="context">Tab A: Causal Q·K·V + FFN</button>
            <button type="button" class="llm-seg__btn" data-tab="layers">Tab B: Layer Stack &amp; KV Sharing</button>
          </div>
        </div>
      </div>
      <div data-role="body"></div>
    `;

    const bodyEl = el.querySelector('[data-role="body"]');

    const stopTimers = () => {
      if (scanTimer) {
        clearInterval(scanTimer);
        scanTimer = null;
      }
      if (cascadeTimer) {
        clearInterval(cascadeTimer);
        cascadeTimer = null;
      }
      scanSubphase = -1;
      layerSubphase = -1;
    };

    const cacheAnimNodes = () => {
      bezParticles = Array.from(bodyEl.querySelectorAll("[data-anim-bez]")).map((node) => {
        const [x0, y0, cx, cy, x1, y1] = node.dataset.animBez.split(",").map(Number);
        return {
          node,
          x0,
          y0,
          cx,
          cy,
          x1,
          y1,
          dur: Number(node.dataset.dur) || 1000,
          phase: Number(node.dataset.phase) || 0,
        };
      });

      polyParticles = Array.from(bodyEl.querySelectorAll("[data-anim-pts]")).map((node) => {
        const pts = node.dataset.animPts.split(";").map((p) => p.split(",").map(Number));
        const segs = [];
        let total = 0;
        for (let i = 0; i < pts.length - 1; i++) {
          const dx = pts[i + 1][0] - pts[i][0];
          const dy = pts[i + 1][1] - pts[i][1];
          const len = Math.hypot(dx, dy);
          segs.push({ x0: pts[i][0], y0: pts[i][1], dx, dy, len });
          total += len;
        }
        return {
          node,
          segs,
          total: total || 1,
          dur: Number(node.dataset.dur) || 1000,
          phase: Number(node.dataset.phase) || 0,
        };
      });

      flowWires = Array.from(bodyEl.querySelectorAll("[data-flow-wire]")).map((node) => ({
        node,
        speed: Number(node.dataset.flowWire) || 26,
      }));
    };

    const stepParticles = (now) => {
      for (let i = 0; i < bezParticles.length; i++) {
        const b = bezParticles[i];
        const t = ((now / b.dur + b.phase) % 1 + 1) % 1;
        const u = 1 - t;
        const px = u * u * b.x0 + 2 * u * t * b.cx + t * t * b.x1;
        const py = u * u * b.y0 + 2 * u * t * b.cy + t * t * b.y1;
        b.node.setAttribute("cx", px.toFixed(1));
        b.node.setAttribute("cy", py.toFixed(1));
      }
      for (let i = 0; i < polyParticles.length; i++) {
        const p = polyParticles[i];
        const t = ((now / p.dur + p.phase) % 1 + 1) % 1;
        let dist = t * p.total;
        let px = p.segs[0] ? p.segs[0].x0 : 0;
        let py = p.segs[0] ? p.segs[0].y0 : 0;
        for (let s = 0; s < p.segs.length; s++) {
          const seg = p.segs[s];
          if (dist <= seg.len || s === p.segs.length - 1) {
            const u = seg.len > 0 ? dist / seg.len : 0;
            px = seg.x0 + seg.dx * u;
            py = seg.y0 + seg.dy * u;
            break;
          }
          dist -= seg.len;
        }
        p.node.setAttribute("cx", px.toFixed(1));
        p.node.setAttribute("cy", py.toFixed(1));
      }
      for (let i = 0; i < flowWires.length; i++) {
        const fw = flowWires[i];
        const offset = -((now / fw.speed) % 24);
        fw.node.setAttribute("stroke-dashoffset", offset.toFixed(1));
      }
    };

    const ensureRaf = () => {
      if (rafId || typeof window.requestAnimationFrame !== "function") return;
      if (!isVisible()) return;
      const tick = (now) => {
        if (!isVisible()) {
          rafId = 0;
          return;
        }
        stepParticles(now);
        rafId = window.requestAnimationFrame(tick);
      };
      rafId = window.requestAnimationFrame(tick);
    };

    if (panelEl && typeof MutationObserver === "function") {
      const obs = new MutationObserver(() => {
        if (isVisible()) ensureRaf();
      });
      obs.observe(panelEl, { attributes: true, attributeFilter: ["hidden"] });
    }

    const getWeightsForWord = (s, idx) => {
      if (s.weights[idx]) return s.weights[idx];
      const arr = new Array(s.words.length).fill(0);
      if (idx === 0) {
        arr[0] = 1;
        return arr;
      }
      for (let i = 0; i <= idx; i++) {
        arr[i] = i === idx ? 0.35 : 0.65 / idx;
      }
      return arr;
    };

    const render = () => {
      el.querySelectorAll("[data-tab]").forEach((b) => {
        const isActive = b.dataset.tab === tab;
        b.classList.toggle("llm-seg__btn--active", isActive);
        b.setAttribute("aria-pressed", String(isActive));
      });

      if (tab === "context") {
        const s = ATTN_SENTENCES[sentIdx];
        const w = getWeightsForWord(s, wordIdx);
        const spacing = 58.4;
        const xStart = 48;
        const svgWidth = 680;
        const focusX = xStart + wordIdx * spacing;
        const attnHot = scanSubphase === 0;
        const ffnHot = scanSubphase === 1;

        const ranked = w
          .map((weight, i) => ({ weight, i, word: s.words[i] }))
          .filter((d) => d.i <= wordIdx && d.weight > 0)
          .sort((a, b) => b.weight - a.weight);
        const topTwo = ranked.slice(0, 2);
        const topArcSet = new Set(
          ranked
            .filter((d) => d.i < wordIdx && d.weight >= 0.14)
            .slice(0, 2)
            .map((d) => d.i)
        );

        const arcs = w
          .map((weight, i) => {
            if (weight < 0.02 || i >= wordIdx) return "";
            const x1 = focusX;
            const x2 = xStart + i * spacing;
            const midX = (x1 + x2) / 2;
            const arcHeight = Math.min(78, 22 + Math.abs(wordIdx - i) * 7.6);
            const ctrlY = 116 - arcHeight;
            const strokeW = (1.1 + weight * 6.2 + (attnHot ? 0.6 : 0)).toFixed(1);
            const pathD = `M ${x2} 116 Q ${midX} ${ctrlY} ${x1} 116`;
            const durMs = Math.round(Math.max(1160, 3100 - weight * 2100));
            const pRadius = Math.min(4.8, 2.3 + weight * 3.5).toFixed(1);
            const peakY = 116 - arcHeight * 0.5;
            const badge = topArcSet.has(i)
              ? `<g pointer-events="none">
                  <rect x="${midX - 17}" y="${peakY - 8}" width="34" height="15" rx="7.5"
                    fill="var(--gemma-surface)" stroke="var(--gemma-accent)" stroke-width="1.1" />
                  <text x="${midX}" y="${peakY + 2.8}" text-anchor="middle" font-size="8.6" font-weight="700"
                    font-family="monospace" fill="var(--gemma-accent)">${Math.round(weight * 100)}%</text>
                </g>`
              : "";
            return `
              <path d="${pathD}" fill="none" stroke="var(--gemma-accent)" stroke-width="${strokeW}"
                stroke-opacity="${Math.min(0.96, 0.26 + weight)}" />
              ${
                weight >= 0.05
                  ? `<circle r="${pRadius}" fill="var(--gemma-accent-fill)"
                      data-anim-bez="${x2},116,${midX},${ctrlY},${x1},116" data-dur="${durMs}" data-phase="${((i * 0.19) % 1).toFixed(2)}" />`
                  : ""
              }
              ${badge}
            `;
          })
          .join("");

        const nodes = s.words
          .map((word, i) => {
            const x = xStart + i * spacing;
            const weight = w[i] || 0;
            const isFocus = i === wordIdx;
            const isMasked = i > wordIdx;
            const isStrong = !isFocus && !isMasked && weight >= 0.16;
            const barW = Math.max(0, Math.min(44, Math.round(44 * weight)));
            const liftY = isFocus && scanTimer ? -3 : 0;
            const topTag = isFocus
              ? `<rect x="${x - 9}" y="109" width="18" height="12" rx="3.5" fill="var(--gemma-accent-fill)" />
                 <text x="${x}" y="118" text-anchor="middle" font-size="8" font-weight="700" font-family="monospace" fill="var(--gemma-white)">Q</text>`
              : !isMasked && weight >= 0.05
                ? `<rect x="${x - 11}" y="110" width="22" height="11" rx="3.5" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-cat-tool-border)" stroke-width="0.9" />
                   <text x="${x}" y="118" text-anchor="middle" font-size="7" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">K·V</text>`
                : "";
            return `
              <g data-word="${i}" style="cursor:pointer" ${isMasked ? 'opacity="0.42"' : ""} ${liftY ? `transform="translate(0, ${liftY})"` : ""}>
                <rect x="${x - 27}" y="120" width="54" height="44" rx="6"
                  fill="${isFocus ? "var(--gemma-accent-fill)" : isStrong ? "var(--gemma-cat-turn-bg)" : "var(--gemma-surface-sunken)"}"
                  stroke="${isFocus || isStrong ? "var(--gemma-accent)" : "var(--gemma-border-strong)"}"
                  stroke-width="${isFocus ? "1.9" : "1.1"}"
                  ${isMasked ? 'stroke-dasharray="3 2"' : ""} />
                ${topTag}
                <text x="${x}" y="138" text-anchor="middle" font-size="10.8" font-weight="${isFocus || isStrong ? "700" : "500"}"
                  fill="${isFocus ? "var(--gemma-white)" : "var(--md-default-fg-color)"}">${escapeHtml(word)}</text>
                ${
                  !isMasked
                    ? `<rect x="${x - 22}" y="145" width="44" height="4" rx="2" fill="${isFocus ? "var(--gemma-white)" : "var(--gemma-border)"}" ${isFocus ? 'fill-opacity="0.28"' : ""} />
                       <rect x="${x - 22}" y="145" width="${barW}" height="4" rx="2" fill="${isFocus ? "var(--gemma-white)" : "var(--gemma-accent-fill)"}" />`
                    : ""
                }
                <text x="${x}" y="158" text-anchor="middle" font-size="8.5" font-family="monospace"
                  fill="${isFocus ? "var(--gemma-white)" : "var(--gemma-text-muted)"}">${isMasked ? "∅ mask" : Math.round(weight * 100) + "%"}</text>
              </g>
            `;
          })
          .join("");

        const vMixRows = topTwo
          .map((item, rIdx) => {
            const ry = 240 + rIdx * 24;
            const bw = Math.max(4, Math.round(54 * item.weight));
            return `
              <text x="36" y="${ry}" font-size="8.8" font-family="monospace" font-weight="600" fill="var(--md-default-fg-color)">${escapeHtml(item.word.slice(0, 9))}</text>
              <rect x="96" y="${ry - 7}" width="54" height="7" rx="3.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border)" stroke-width="0.8" />
              <rect x="96" y="${ry - 7}" width="${bw}" height="7" rx="3.5" fill="var(--gemma-accent-fill)" />
              <text x="182" y="${ry}" text-anchor="end" font-size="8.5" font-family="monospace" font-weight="700" fill="var(--gemma-accent)">${Math.round(item.weight * 100)}%</text>
            `;
          })
          .join("");

        const gatePath = "M 298 247 L 312 226 L 446 226 L 458 247";
        const upPath = "M 298 247 L 312 268 L 446 268 L 458 247";
        const downPath = "M 469 247 L 618 247";

        if (!bodyEl.querySelector('[data-role="attn-causal-stage"]')) {
          bodyEl.innerHTML = `
            <div class="llm-toolbar" data-role="attn-causal-toolbar" style="justify-content:space-between;margin-bottom:var(--gemma-space-3);">
              <div class="llm-toolbar__group" role="group" aria-label="Select sentence scenario">
                <span class="llm-toolbar__label">Scenario</span>
                ${ATTN_SENTENCES.map(
                  (item, idx) =>
                    `<button type="button" class="llm-scenario-btn" data-sent="${idx}">${escapeHtml(item.label)}</button>`
                ).join("")}
              </div>
              <button type="button" class="llm-action-btn" data-action="auto-scan"></button>
            </div>
            <div data-role="attn-causal-stage"></div>
          `;
        }

        bodyEl.querySelectorAll('[data-role="attn-causal-toolbar"] [data-sent]').forEach((btn) => {
          const isAct = Number(btn.dataset.sent) === sentIdx;
          btn.classList.toggle("llm-scenario-btn--active", isAct);
          btn.setAttribute("aria-pressed", String(isAct));
        });
        const scanBtn = bodyEl.querySelector('[data-action="auto-scan"]');
        if (scanBtn) {
          scanBtn.classList.toggle("llm-action-btn--running", Boolean(scanTimer));
          scanBtn.classList.toggle("llm-btn--primary", !scanTimer);
          const nextScanLabel = scanTimer ? "⏸ Stop" : "▶ Auto-Scan";
          if (scanBtn.textContent !== nextScanLabel) scanBtn.textContent = nextScanLabel;
        }

        const causalStageEl = bodyEl.querySelector('[data-role="attn-causal-stage"]');
        causalStageEl.innerHTML = `
          <svg class="llm-svg-stage" viewBox="0 0 ${svgWidth} 318" aria-label="Causal self-attention and GeGLU FFN dataflow">
            <text x="18" y="22" font-size="9.2" font-weight="700" font-family="monospace" fill="${attnHot ? "var(--gemma-accent)" : "var(--gemma-text-muted)"}">SUBLAYER A · CAUSAL SELF-ATTENTION (softmax(QKᵀ/√dₖ + M))</text>
            <g transform="translate(446, 12)" font-size="8.5" font-family="monospace">
              <rect x="0" y="0" width="13" height="11" rx="3" fill="var(--gemma-accent-fill)" />
              <text x="17" y="9" fill="var(--gemma-text-secondary)" font-weight="600">Query</text>
              <rect x="58" y="0" width="19" height="11" rx="3" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-cat-tool-border)" />
              <text x="81" y="9" fill="var(--gemma-text-secondary)" font-weight="600">Past K·V</text>
              <rect x="136" y="0" width="15" height="11" rx="3" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" stroke-dasharray="2 1.5" />
              <text x="155" y="9" fill="var(--gemma-text-muted)">Masked</text>
            </g>

            ${arcs}
            ${nodes}

            <!-- Tap wire from active token into Sublayer A output + Sublayer B GeGLU FFN -->
            <path d="M ${focusX} 164 L ${focusX} 178 L 108 178 L 108 202" fill="none" stroke="var(--gemma-accent)" stroke-width="1.7" stroke-dasharray="4 3" data-flow-wire="44" />
            <circle r="3.2" fill="var(--gemma-accent-fill)" data-anim-pts="${focusX},164;${focusX},178;108,178;108,202" data-dur="1440" data-phase="0" />

            <!-- Bottom circuit container -->
            <rect x="16" y="192" width="648" height="112" rx="9" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border)" />

            <!-- Box 1: Weighted Value Mix -->
            <rect x="28" y="202" width="164" height="90" rx="6" fill="var(--gemma-surface)"
              stroke="${attnHot ? "var(--gemma-accent)" : "var(--gemma-cat-turn-border)"}" stroke-width="${attnHot ? "1.9" : "1.1"}" />
            <text x="36" y="218" font-size="8.6" font-weight="700" font-family="monospace" fill="var(--gemma-accent)">Σ αᵢ·Vᵢ → o_proj (${escapeHtml(s.words[wordIdx])})</text>
            ${vMixRows}

            <!-- Residual Add #1 + RMSNorm -->
            <line x1="192" y1="247" x2="298" y2="247" stroke="var(--gemma-accent)" stroke-width="1.5" />
            <circle r="2.8" fill="var(--gemma-accent-fill)" data-anim-pts="192,247;298,247" data-dur="1240" data-phase="0.35" />
            <circle cx="220" cy="247" r="10.5" fill="var(--gemma-surface)" stroke="var(--gemma-accent)" stroke-width="1.4" />
            <text x="220" y="250.8" text-anchor="middle" font-size="11" font-weight="700" fill="var(--gemma-accent)">⊕</text>

            <!-- Residual skip over FFN -->
            <path d="M 236 247 L 236 196 L 602 196 L 602 236" fill="none" stroke="var(--gemma-border-strong)" stroke-width="1.2" stroke-dasharray="3 2" data-flow-wire="68" />

            <rect x="244" y="234" width="54" height="26" rx="5" fill="var(--gemma-surface)" stroke="var(--gemma-border-strong)" />
            <text x="271" y="250" text-anchor="middle" font-size="8.5" font-weight="700" font-family="monospace" fill="var(--gemma-text-secondary)">RMSNorm</text>

            <!-- Sublayer B: GeGLU FFN Circuit -->
            <rect x="304" y="202" width="280" height="90" rx="7" fill="var(--gemma-surface)"
              stroke="${ffnHot ? "var(--gemma-status-success)" : "var(--gemma-cat-tool-border)"}" stroke-width="${ffnHot ? "2" : "1.1"}" />
            <text x="574" y="215" text-anchor="end" font-size="8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">SUBLAYER B · GeGLU FFN</text>

            <path d="${gatePath}" fill="none" stroke="var(--gemma-cat-think-border)" stroke-width="1.6" />
            <path d="${upPath}" fill="none" stroke="var(--gemma-cat-tool-border)" stroke-width="1.6" />
            <path d="${downPath}" fill="none" stroke="var(--gemma-status-success)" stroke-width="1.7" />

            <rect x="316" y="214" width="68" height="24" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="350" y="229" text-anchor="middle" font-size="8.5" font-weight="600" font-family="monospace" fill="var(--md-default-fg-color)">gate_proj</text>

            <rect x="392" y="214" width="44" height="24" rx="4.5" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-cat-think-border)" />
            <text x="414" y="229" text-anchor="middle" font-size="8.5" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">GELU</text>

            <rect x="316" y="256" width="120" height="24" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="376" y="271" text-anchor="middle" font-size="8.5" font-weight="600" font-family="monospace" fill="var(--md-default-fg-color)">up_proj</text>

            <circle cx="458" cy="247" r="10.5" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-status-success)" stroke-width="1.5" />
            <text x="458" y="250.8" text-anchor="middle" font-size="11" font-weight="700" fill="var(--gemma-cat-tool-fg)">⊙</text>

            <rect x="478" y="234" width="96" height="26" rx="5" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-cat-tool-border)" />
            <text x="526" y="250" text-anchor="middle" font-size="8.8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">down_proj</text>

            <!-- Residual Add #2 + Output vector -->
            <circle cx="602" cy="247" r="10.5" fill="var(--gemma-surface)" stroke="var(--gemma-status-success)" stroke-width="1.4" />
            <text x="602" y="250.8" text-anchor="middle" font-size="11" font-weight="700" fill="var(--gemma-cat-tool-fg)">⊕</text>
            <rect x="618" y="232" width="40" height="30" rx="6" fill="${ffnHot ? "var(--gemma-status-success)" : "var(--gemma-accent-fill)"}" />
            <text x="638" y="250.5" text-anchor="middle" font-size="9.2" font-weight="700" font-family="monospace" fill="var(--gemma-white)">h'ᵢ</text>

            <!-- Live GeGLU particles rendered on top of boxes -->
            <circle r="3.2" fill="var(--gemma-status-warning)" data-anim-pts="298,247;312,226;446,226;458,247" data-dur="1840" data-phase="0" />
            <circle r="3.2" fill="var(--gemma-status-success)" data-anim-pts="298,247;312,268;446,268;458,247" data-dur="1840" data-phase="0" />
            <circle r="3.4" fill="var(--gemma-accent-fill)" data-anim-pts="469,247;618,247" data-dur="1440" data-phase="0.5" />
          </svg>
        `;
      } else {
        const m = GEMMA_MODELS_ARCH[modelIdx];
        if (selectedLayer > m.layers) selectedLayer = m.layers;
        const isGlobal = selectedLayer % m.period === 0;
        const hasKvShare = m.kvSharedLayers > 0;
        const firstKvShared = hasKvShare ? m.layers - m.kvSharedLayers + 1 : Infinity;
        const isKvShared = selectedLayer >= firstKvShared;
        const localSrc = hasKvShare ? firstKvShared - 2 : 0;
        const globalSrc = hasKvShare ? firstKvShared - 1 : 0;
        const activeSrc = isGlobal ? globalSrc : localSrc;
        const isDoubleWide = Boolean(m.doubleWideKvShared && isKvShared);
        const attnPhaseHot = layerSubphase === 0;
        const ffnPhaseHot = layerSubphase === 1;

        // --- SVG 1: Layer Stack Strip + KV Sharing / K=V Topology ---
        const padL = 16;
        const trackW = 648;
        const step = trackW / m.layers;
        const colW = Math.max(6.8, Math.min(14, step - 4.2));
        const cx = (lNum) => padL + (lNum - 0.5) * step;

        let kvTopologySvg = "";
        if (hasKvShare) {
          const zoneX1 = cx(firstKvShared) - step * 0.46;
          const zoneX2 = cx(m.layers) + step * 0.46;
          const bgArcs = [];
          for (let l = firstKvShared; l <= m.layers; l++) {
            const g = l % m.period === 0;
            const sL = g ? globalSrc : localSrc;
            if (g && l !== selectedLayer) {
              const x1 = cx(sL);
              const x2 = cx(l);
              const mid = (x1 + x2) / 2;
              bgArcs.push(
                `<path d="M ${x1} 72 Q ${mid} 38 ${x2} 72" fill="none"
                  stroke="var(--gemma-accent)" stroke-width="1.1" stroke-opacity="0.28" stroke-dasharray="2.5 2" />
                 <circle r="2.2" fill="var(--gemma-accent-fill)" fill-opacity="0.55"
                  data-anim-bez="${x1},72,${mid},38,${x2},72" data-dur="2700" data-phase="${((l * 0.17) % 1).toFixed(2)}" />`
              );
            }
          }

          const heroTarget = isKvShared ? selectedLayer : m.layers;
          const heroGlobal = heroTarget % m.period === 0;
          const heroSrc = heroGlobal ? globalSrc : localSrc;
          const hx1 = cx(heroSrc);
          const hx2 = cx(heroTarget);
          const hMid = (hx1 + hx2) / 2;
          const hPath = `M ${hx1} 72 Q ${hMid} 28 ${hx2} 72`;
          const badgeW = 82;
          const badgeX = Math.max(padL + badgeW / 2, Math.min(680 - padL - badgeW / 2, hMid));

          kvTopologySvg = `
            <rect x="${zoneX1}" y="68" width="${zoneX2 - zoneX1}" height="68" rx="6"
              fill="var(--gemma-cat-think-bg)" fill-opacity="0.38" stroke="var(--gemma-cat-think-border)" stroke-dasharray="3 2" />
            ${bgArcs.join("")}
            <path d="${hPath}" fill="none" stroke="var(--gemma-status-warning)" stroke-width="2.2" stroke-dasharray="5 3" data-flow-wire="40" />
            <circle r="3.6" fill="var(--gemma-status-warning)" data-anim-bez="${hx1},72,${hMid},28,${hx2},72" data-dur="1760" data-phase="0" />
            <circle r="3" fill="var(--gemma-accent-fill)" data-anim-bez="${hx1},72,${hMid},28,${hx2},72" data-dur="1760" data-phase="0.5" />
            <circle cx="${cx(localSrc)}" cy="72" r="3.4" fill="var(--gemma-status-warning)" />
            <circle cx="${cx(globalSrc)}" cy="72" r="3.8" fill="var(--gemma-accent-fill)" stroke="var(--gemma-white)" stroke-width="1" />
            <rect x="${badgeX - badgeW / 2}" y="27" width="${badgeW}" height="15" rx="7.5" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-status-warning)" />
            <text x="${badgeX}" y="37.2" text-anchor="middle" font-size="8.2" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">L${heroSrc} ↷ L${heroTarget} KV</text>
          `;
        } else if (m.kEqV) {
          const globalXs = [];
          for (let l = m.period; l <= m.layers; l += m.period) {
            globalXs.push({ l, x: cx(l) });
          }
          const railX1 = globalXs[0].x;
          const railX2 = globalXs[globalXs.length - 1].x;
          const activeGlobX = isGlobal ? cx(selectedLayer) : globalXs[globalXs.length - 1].x;
          kvTopologySvg = `
            <line x1="${railX1}" y1="42" x2="${railX2}" y2="42" stroke="var(--gemma-cat-multi-border)" stroke-width="1.7" stroke-dasharray="4 3" data-flow-wire="48" />
            ${globalXs
              .map(
                (g) => `
                  <line x1="${g.x}" y1="42" x2="${g.x}" y2="71" stroke="${g.l === selectedLayer ? "var(--gemma-accent)" : "var(--gemma-cat-multi-border)"}" stroke-width="${g.l === selectedLayer ? "2.1" : "1.3"}" />
                  <circle cx="${g.x}" cy="42" r="${g.l === selectedLayer ? "3.8" : "2.6"}" fill="${g.l === selectedLayer ? "var(--gemma-accent-fill)" : "var(--gemma-cat-multi-fg)"}" />
                `
              )
              .join("")}
            <circle r="3.2" fill="var(--gemma-accent-fill)" data-anim-pts="${activeGlobX},42;${activeGlobX},71" data-dur="1360" data-phase="0" />
            <rect x="${(railX1 + railX2) / 2 - 86}" y="31" width="172" height="16" rx="8" fill="var(--gemma-cat-multi-bg)" stroke="var(--gemma-cat-multi-border)" />
            <text x="${(railX1 + railX2) / 2}" y="41.8" text-anchor="middle" font-size="8.2" font-weight="700" font-family="monospace" fill="var(--gemma-cat-multi-fg)">Unified K = V · ${m.globalGqa} Global GQA</text>
          `;
        }

        const layerColsSvg = Array.from({ length: m.layers }, (_, i) => {
          const lNum = i + 1;
          const glob = lNum % m.period === 0;
          const kvSh = lNum >= firstKvShared;
          const isSrc = hasKvShare && (lNum === localSrc || lNum === globalSrc);
          const isAct = lNum === selectedLayer;
          const isFutureInCascade = Boolean(cascadeTimer && lNum > selectedLayer);
          const x = cx(lNum) - colW / 2;
          const attnY = glob ? 72 : 82;
          const attnH = glob ? 30 : 20;
          const ffnH = m.doubleWideKvShared && kvSh ? 26 : 20;
          const lift = isAct && cascadeTimer ? -3.5 : 0;
          const showTick =
            lNum === 1 ||
            lNum === m.layers ||
            lNum === firstKvShared ||
            isAct ||
            (m.layers <= 48 ? glob : lNum % 12 === 0);

          return `
            <g data-layer="${lNum}" style="cursor:pointer" ${isFutureInCascade ? 'opacity="0.45"' : ""} ${lift ? `transform="translate(0, ${lift})"` : ""}>
              <rect x="${cx(lNum) - step / 2}" y="66" width="${step}" height="88" fill="transparent" />
              ${
                isAct
                  ? `<rect x="${x - 2.4}" y="69" width="${colW + 4.8}" height="66" rx="4.5" fill="none"
                      stroke="${ffnPhaseHot ? "var(--gemma-status-success)" : "var(--gemma-accent)"}" stroke-width="2.1" />`
                  : ""
              }
              <rect x="${x}" y="${attnY}" width="${colW}" height="${attnH}" rx="2.5"
                fill="${glob ? "var(--gemma-accent-fill)" : kvSh ? "var(--gemma-cat-think-bg)" : "var(--gemma-surface-sunken)"}"
                stroke="${isAct && attnPhaseHot ? "var(--gemma-accent)" : isSrc ? "var(--gemma-status-warning)" : glob ? "var(--gemma-accent)" : kvSh ? "var(--gemma-cat-think-border)" : "var(--gemma-border-strong)"}"
                stroke-width="${isAct && attnPhaseHot ? "1.9" : isSrc ? "1.6" : "1.1"}"
                ${kvSh ? 'stroke-dasharray="2.2 1.4"' : ""} />
              <rect x="${x}" y="106" width="${colW}" height="${ffnH}" rx="2.5"
                fill="${isAct && ffnPhaseHot ? "var(--gemma-status-success)" : m.id === "26b" ? "var(--gemma-cat-multi-bg)" : m.doubleWideKvShared && kvSh ? "var(--gemma-status-success)" : "var(--gemma-cat-tool-bg)"}"
                fill-opacity="${m.doubleWideKvShared && kvSh && !(isAct && ffnPhaseHot) ? "0.78" : "1"}"
                stroke="${isAct && ffnPhaseHot ? "var(--gemma-status-success)" : m.id === "26b" ? "var(--gemma-cat-multi-border)" : "var(--gemma-cat-tool-border)"}" />
              ${
                showTick
                  ? `<text x="${cx(lNum)}" y="149" text-anchor="middle" font-size="${isAct ? "9" : "7.6"}" font-weight="${isAct ? "700" : "500"}"
                      font-family="monospace" fill="${isAct ? "var(--gemma-accent)" : "var(--gemma-text-muted)"}">L${lNum}</text>`
                  : ""
              }
            </g>
          `;
        }).join("");

        // --- SVG 2A: Sublayer A (Attention & KV Cache Circuit) ---
        const headDim = isGlobal ? 512 : 256;
        const kvHeads = isGlobal ? m.globalKvHeads : m.localKvHeads;
        const gqaRatio = isGlobal ? m.globalGqa : m.localGqa;
        const qPath = "M 32 118 L 40 66 L 210 66";
        let kvRowsSvg = "";
        let kvParticlesSvg = "";

        if (m.kEqV && isGlobal) {
          const kPath = "M 32 118 L 210 118";
          const vEqKPath = "M 140 118 L 140 150 L 210 150";
          kvRowsSvg = `
            <path d="${kPath}" fill="none" stroke="var(--gemma-cat-multi-border)" stroke-width="1.6" />
            <path d="${vEqKPath}" fill="none" stroke="var(--gemma-accent)" stroke-width="1.7" stroke-dasharray="4 2.5" data-flow-wire="40" />
            <rect x="46" y="103" width="90" height="30" rx="4.5" fill="var(--gemma-cat-multi-bg)" stroke="var(--gemma-cat-multi-border)" stroke-width="1.4" />
            <text x="91" y="116" text-anchor="middle" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--gemma-cat-multi-fg)">k_proj (K=V)</text>
            <text x="91" y="127.5" text-anchor="middle" font-size="7.5" font-family="monospace" fill="var(--gemma-text-secondary)">${kvHeads}KV · ${gqaRatio} GQA</text>

            <rect x="148" y="106" width="52" height="24" rx="4.5" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-cat-turn-border)" />
            <text x="174" y="121" text-anchor="middle" font-size="8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-turn-fg)">p-RoPE</text>

            <rect x="152" y="141" width="46" height="18" rx="9" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-cat-tool-border)" />
            <text x="175" y="153" text-anchor="middle" font-size="7.8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">V ← K</text>

            <g opacity="0.45">
              <line x1="32" y1="118" x2="46" y2="177" stroke="var(--gemma-border-strong)" stroke-dasharray="2 2" />
              <rect x="46" y="163" width="90" height="28" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" stroke-dasharray="3 2" />
              <text x="91" y="180" text-anchor="middle" font-size="8" font-family="monospace" fill="var(--gemma-text-muted)">v_proj ✕ (−50%)</text>
            </g>
          `;
          kvParticlesSvg = `
            <circle r="3" fill="var(--gemma-cat-multi-fg)" data-anim-pts="32,118;210,118" data-dur="1840" data-phase="0" />
            <circle r="3" fill="var(--gemma-status-success)" data-anim-pts="140,118;140,150;210,150" data-dur="1520" data-phase="0.45" />
          `;
        } else if (isKvShared) {
          kvRowsSvg = `
            <g opacity="0.4">
              <rect x="46" y="104" width="60" height="26" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" stroke-dasharray="3 2" />
              <text x="76" y="120" text-anchor="middle" font-size="7.8" font-family="monospace" fill="var(--gemma-text-muted)">k_proj ✕</text>
              <rect x="46" y="158" width="60" height="26" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" stroke-dasharray="3 2" />
              <text x="76" y="174" text-anchor="middle" font-size="7.8" font-family="monospace" fill="var(--gemma-text-muted)">v_proj ✕</text>
            </g>
            <path d="M 198 118 L 210 118" fill="none" stroke="var(--gemma-status-warning)" stroke-width="1.9" />
            <path d="M 198 150 L 210 150" fill="none" stroke="var(--gemma-status-warning)" stroke-width="1.9" />
            <rect x="114" y="102" width="84" height="84" rx="6" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-status-warning)" stroke-width="1.6" />
            <text x="156" y="121" text-anchor="middle" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">KV Shared</text>
            <text x="156" y="138" text-anchor="middle" font-size="8.8" font-weight="700" font-family="monospace" fill="var(--gemma-accent)">← Layer L${activeSrc}</text>
            <text x="156" y="155" text-anchor="middle" font-size="7.6" font-family="monospace" fill="var(--gemma-text-secondary)">${kvHeads}KV · ${headDim}d</text>
            <text x="156" y="172" text-anchor="middle" font-size="7.4" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">0 new KV RAM</text>
          `;
          kvParticlesSvg = `
            <circle r="3" fill="var(--gemma-status-warning)" data-anim-pts="198,118;210,118" data-dur="1360" data-phase="0" />
            <circle r="3" fill="var(--gemma-status-warning)" data-anim-pts="198,150;210,150" data-dur="1360" data-phase="0.5" />
          `;
        } else {
          const kPath = "M 32 118 L 210 118";
          const vPath = "M 32 118 L 40 172 L 210 172";
          const isAnchor = hasKvShare && (selectedLayer === localSrc || selectedLayer === globalSrc);
          kvRowsSvg = `
            <path d="${kPath}" fill="none" stroke="var(--gemma-border-strong)" stroke-width="1.5" />
            <path d="${vPath}" fill="none" stroke="var(--gemma-border-strong)" stroke-width="1.5" />
            <rect x="46" y="103" width="90" height="29" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="${isAnchor ? "var(--gemma-status-warning)" : "var(--gemma-border-strong)"}" />
            <text x="91" y="116" text-anchor="middle" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">k_proj</text>
            <text x="91" y="127" text-anchor="middle" font-size="7.4" font-family="monospace" fill="var(--gemma-text-muted)">${kvHeads}KV · ${gqaRatio} GQA</text>

            <rect x="144" y="106" width="54" height="23" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="171" y="120.5" text-anchor="middle" font-size="7.8" font-weight="600" font-family="monospace" fill="var(--gemma-text-secondary)">${isGlobal ? "p-RoPE" : "RoPE"}</text>

            <rect x="46" y="157" width="90" height="29" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="${isAnchor ? "var(--gemma-status-warning)" : "var(--gemma-border-strong)"}" />
            <text x="91" y="170" text-anchor="middle" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">v_proj</text>
            <text x="91" y="181" text-anchor="middle" font-size="7.4" font-family="monospace" fill="var(--gemma-text-muted)">${kvHeads}KV · K ≠ V</text>
            ${
              isAnchor
                ? `<rect x="140" y="164" width="66" height="17" rx="8.5" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-status-warning)" />
                   <text x="173" y="175" text-anchor="middle" font-size="7.2" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">↷ L${firstKvShared}–L${m.layers}</text>`
                : ""
            }
          `;
          kvParticlesSvg = `
            <circle r="2.8" fill="var(--gemma-accent-fill)" data-anim-pts="32,118;210,118" data-dur="1880" data-phase="0.15" />
            <circle r="2.8" fill="var(--gemma-status-success)" data-anim-pts="32,118;40,172;210,172" data-dur="1960" data-phase="0.35" />
          `;
        }

        const sublayerASvg = `
          <svg class="llm-svg-stage" viewBox="0 0 334 210" style="${attnPhaseHot ? "border-color:var(--gemma-accent);box-shadow:var(--gemma-shadow-sm);" : ""}" aria-label="Sublayer A Attention and KV Cache circuit">
            <text x="12" y="21" font-size="9.6" font-weight="700" font-family="monospace" fill="${attnPhaseHot ? "var(--gemma-accent)" : "var(--md-default-fg-color)"}">L${selectedLayer} · ${isGlobal ? "🌐 Global Attn" : "⚡ Local Attn"}</text>
            <text x="322" y="21" text-anchor="end" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--gemma-accent)">${isGlobal ? m.globalCtxShort + " · p-RoPE 25%" : m.windowShort + " · RoPE 10K"}</text>
            <line x1="10" y1="31" x2="324" y2="31" stroke="var(--gemma-border)" />

            <!-- Residual bypass -->
            <path d="M 21 106 L 21 41 L 318 41 L 318 108" fill="none" stroke="var(--gemma-border-strong)" stroke-width="1.1" stroke-dasharray="3 2" data-flow-wire="68" />

            <!-- Input h -->
            <rect x="10" y="106" width="22" height="24" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="21" y="121" text-anchor="middle" font-size="9" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">h</text>

            <!-- Q row -->
            <path d="${qPath}" fill="none" stroke="var(--gemma-accent)" stroke-width="1.5" />
            <rect x="46" y="51" width="90" height="29" rx="4.5" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-accent)" />
            <text x="91" y="64" text-anchor="middle" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--gemma-cat-turn-fg)">q_proj</text>
            <text x="91" y="75" text-anchor="middle" font-size="7.4" font-family="monospace" fill="var(--gemma-text-secondary)">${m.qHeads}Q × ${headDim}d</text>

            <rect x="144" y="54" width="54" height="23" rx="4.5" fill="var(--gemma-cat-turn-bg)" stroke="var(--gemma-cat-turn-border)" />
            <text x="171" y="68.5" text-anchor="middle" font-size="7.8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-turn-fg)">${isGlobal ? "p-RoPE" : "RoPE"}</text>

            ${kvRowsSvg}

            <!-- Attention Core + o_proj + Residual -->
            <rect x="210" y="54" width="56" height="130" rx="6" fill="var(--gemma-accent-fill)" />
            <text x="238" y="102" text-anchor="middle" font-size="8.8" font-weight="700" font-family="monospace" fill="var(--gemma-white)">QKᵀ</text>
            <text x="238" y="121" text-anchor="middle" font-size="7.6" font-family="monospace" fill="var(--gemma-white)">softmax</text>
            <text x="238" y="140" text-anchor="middle" font-size="8.8" font-weight="700" font-family="monospace" fill="var(--gemma-white)">× V</text>

            <line x1="266" y1="118" x2="310" y2="118" stroke="var(--gemma-accent)" stroke-width="1.5" />
            <rect x="274" y="104" width="32" height="28" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
            <text x="290" y="121" text-anchor="middle" font-size="7.6" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">o_proj</text>

            <circle cx="318" cy="118" r="9" fill="var(--gemma-surface)" stroke="var(--gemma-accent)" stroke-width="1.4" />
            <text x="318" y="121.5" text-anchor="middle" font-size="10" font-weight="700" fill="var(--gemma-accent)">⊕</text>

            <!-- Sublayer A particles rendered on top -->
            <circle r="2.9" fill="var(--gemma-accent-fill)" data-anim-pts="32,118;40,66;210,66" data-dur="1760" data-phase="0" />
            ${kvParticlesSvg}
            <circle r="2.9" fill="var(--gemma-accent-fill)" data-anim-pts="266,118;318,118" data-dur="1240" data-phase="0.6" />
          </svg>
        `;

        // --- SVG 2B: Sublayer B (GeGLU FFN or MoE Circuit) ---
        let sublayerBSvg = "";
        if (m.id === "26b") {
          const activeRouters = new Set();
          for (let r = 0; r < 8; r++) {
            activeRouters.add((selectedLayer * 7 + r * 5) % 32);
          }
          const expertCells = [];
          const routerRays = [];
          const routerParticles = [];
          let rayCount = 0;
          for (let idx = 0; idx < 32; idx++) {
            const col = idx % 8;
            const row = Math.floor(idx / 8);
            const ex = 148 + col * 16;
            const ey = 54 + row * 18;
            const act = activeRouters.has(idx);
            expertCells.push(
              `<rect x="${ex}" y="${ey}" width="13.5" height="13.5" rx="2.5"
                fill="${act ? "var(--gemma-status-success)" : "var(--gemma-surface-sunken)"}"
                stroke="${act ? "var(--gemma-cat-tool-border)" : "var(--gemma-border)"}" />`
            );
            if (act) {
              routerRays.push(
                `<line x1="136" y1="84" x2="${ex}" y2="${ey + 6.8}" stroke="var(--gemma-status-success)" stroke-width="1.1" stroke-opacity="0.7" />`
              );
              if (rayCount % 2 === 0) {
                routerParticles.push(
                  `<circle r="2.5" fill="var(--gemma-status-success)" data-anim-pts="80,120;113,84;${ex + 6.5},${ey + 6.8};276,90;304,116" data-dur="1920" data-phase="${(rayCount * 0.14).toFixed(2)}" />`
                );
              }
              rayCount += 1;
            }
          }
          const sharedPath = "M 80 120 L 94 166 L 298 166 L 312 129";

          sublayerBSvg = `
            <svg class="llm-svg-stage" viewBox="0 0 334 210" style="${ffnPhaseHot ? "border-color:var(--gemma-status-success);box-shadow:var(--gemma-shadow-sm);" : ""}" aria-label="Sublayer B Mixture of Experts circuit">
              <text x="12" y="21" font-size="9.6" font-weight="700" font-family="monospace" fill="${ffnPhaseHot ? "var(--gemma-cat-tool-fg)" : "var(--md-default-fg-color)"}">L${selectedLayer} · GeGLU MoE</text>
              <text x="322" y="21" text-anchor="end" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">8/128 Routed + 1 Shared</text>
              <line x1="10" y1="31" x2="324" y2="31" stroke="var(--gemma-border)" />

              <path d="M 21 108 L 21 41 L 312 41 L 312 110" fill="none" stroke="var(--gemma-border-strong)" stroke-width="1.1" stroke-dasharray="3 2" data-flow-wire="68" />

              <rect x="10" y="108" width="22" height="24" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
              <text x="21" y="123" text-anchor="middle" font-size="9" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">x</text>

              <rect x="38" y="107" width="42" height="26" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
              <text x="59" y="123" text-anchor="middle" font-size="7.5" font-weight="700" font-family="monospace" fill="var(--gemma-text-secondary)">RMSNorm</text>

              <path d="M 80 120 L 90 84" fill="none" stroke="var(--gemma-status-warning)" stroke-width="1.4" />
              <path d="${sharedPath}" fill="none" stroke="var(--gemma-cat-multi-border)" stroke-width="1.6" />

              <rect x="90" y="69" width="46" height="30" rx="4.5" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-status-warning)" />
              <text x="113" y="82.5" text-anchor="middle" font-size="8.2" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">Router</text>
              <text x="113" y="93" text-anchor="middle" font-size="7.2" font-family="monospace" fill="var(--gemma-text-secondary)">Top-8</text>

              ${routerRays.join("")}
              ${expertCells.join("")}
              <text x="212" y="138" text-anchor="middle" font-size="7.4" font-family="monospace" fill="var(--gemma-text-muted)">8 of 128 GeGLU Experts (704d)</text>
              <line x1="276" y1="90" x2="304" y2="116" stroke="var(--gemma-status-success)" stroke-width="1.5" />

              <rect x="148" y="150" width="126" height="32" rx="5" fill="var(--gemma-cat-multi-bg)" stroke="var(--gemma-cat-multi-border)" stroke-width="1.4" />
              <text x="211" y="164" text-anchor="middle" font-size="8" font-weight="700" font-family="monospace" fill="var(--gemma-cat-multi-fg)">1 Shared GeGLU Expert</text>
              <text x="211" y="175.5" text-anchor="middle" font-size="7.2" font-family="monospace" fill="var(--gemma-text-secondary)">Always-On · 2,112d</text>

              <circle cx="312" cy="120" r="9" fill="var(--gemma-surface)" stroke="var(--gemma-status-success)" stroke-width="1.4" />
              <text x="312" y="123.5" text-anchor="middle" font-size="10" font-weight="700" fill="var(--gemma-cat-tool-fg)">⊕</text>

              ${routerParticles.join("")}
              <circle r="3" fill="var(--gemma-cat-multi-fg)" data-anim-pts="80,120;94,166;298,166;312,129" data-dur="1920" data-phase="0.2" />
            </svg>
          `;
        } else {
          const dFfLabel = isDoubleWide ? m.dFfWide : m.dFf;
          const boxH = isDoubleWide ? 34 : 27;
          const gateY = isDoubleWide ? 64 : 69;
          const upY = isDoubleWide ? 142 : 146;
          const ffnGatePath = "M 90 120 L 104 82 L 234 82 L 244 120";
          const ffnUpPath = "M 90 120 L 104 159 L 234 159 L 244 120";
          const ffnDownPath = "M 254 120 L 307 120";

          sublayerBSvg = `
            <svg class="llm-svg-stage" viewBox="0 0 334 210" style="${ffnPhaseHot ? "border-color:var(--gemma-status-success);box-shadow:var(--gemma-shadow-sm);" : ""}" aria-label="Sublayer B GeGLU Feed-Forward circuit">
              <text x="12" y="21" font-size="9.6" font-weight="700" font-family="monospace" fill="${ffnPhaseHot ? "var(--gemma-cat-tool-fg)" : "var(--md-default-fg-color)"}">L${selectedLayer} · GeGLU FFN${isDoubleWide ? " (2× Wide)" : ""}</text>
              <text x="322" y="21" text-anchor="end" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">${m.dModel} → ${dFfLabel} → ${m.dModel}</text>
              <line x1="10" y1="31" x2="324" y2="31" stroke="var(--gemma-border)" />

              <!-- Residual bypass -->
              <path d="M 21 108 L 21 41 L 316 41 L 316 111" fill="none" stroke="var(--gemma-border-strong)" stroke-width="1.1" stroke-dasharray="3 2" data-flow-wire="68" />
              <text x="168" y="52" text-anchor="middle" font-size="7.4" font-family="monospace" fill="var(--gemma-text-muted)">residual skip</text>

              <rect x="10" y="108" width="22" height="24" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
              <text x="21" y="123" text-anchor="middle" font-size="9" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">x</text>

              <line x1="32" y1="120" x2="40" y2="120" stroke="var(--gemma-border-strong)" stroke-width="1.4" />
              <rect x="40" y="107" width="50" height="26" rx="4.5" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
              <text x="65" y="123" text-anchor="middle" font-size="7.8" font-weight="700" font-family="monospace" fill="var(--gemma-text-secondary)">RMSNorm</text>

              <path d="${ffnGatePath}" fill="none" stroke="var(--gemma-cat-think-border)" stroke-width="1.6" />
              <path d="${ffnUpPath}" fill="none" stroke="var(--gemma-cat-tool-border)" stroke-width="1.6" />
              <path d="${ffnDownPath}" fill="none" stroke="var(--gemma-status-success)" stroke-width="1.7" />

              <rect x="106" y="${gateY}" width="72" height="${boxH}" rx="4.5"
                fill="${isDoubleWide ? "var(--gemma-cat-tool-bg)" : "var(--gemma-surface-sunken)"}"
                stroke="${isDoubleWide ? "var(--gemma-status-success)" : "var(--gemma-border-strong)"}" />
              <text x="142" y="${gateY + boxH / 2 + 3}" text-anchor="middle" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">gate_proj</text>

              <rect x="184" y="69" width="42" height="27" rx="4.5" fill="var(--gemma-cat-think-bg)" stroke="var(--gemma-cat-think-border)" />
              <text x="205" y="85.5" text-anchor="middle" font-size="8.2" font-weight="700" font-family="monospace" fill="var(--gemma-cat-think-fg)">GELU</text>

              <rect x="106" y="${upY}" width="120" height="${boxH}" rx="4.5"
                fill="${isDoubleWide ? "var(--gemma-cat-tool-bg)" : "var(--gemma-surface-sunken)"}"
                stroke="${isDoubleWide ? "var(--gemma-status-success)" : "var(--gemma-border-strong)"}" />
              <text x="166" y="${upY + boxH / 2 + 3}" text-anchor="middle" font-size="8.4" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">up_proj (${dFfLabel}d)</text>

              <circle cx="244" cy="120" r="10" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-status-success)" stroke-width="1.5" />
              <text x="244" y="123.5" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--gemma-cat-tool-fg)">⊙</text>

              <rect x="260" y="105" width="44" height="30" rx="4.5" fill="var(--gemma-cat-tool-bg)" stroke="var(--gemma-cat-tool-border)" />
              <text x="282" y="118.5" text-anchor="middle" font-size="7.6" font-weight="700" font-family="monospace" fill="var(--gemma-cat-tool-fg)">down</text>
              <text x="282" y="128.5" text-anchor="middle" font-size="7.2" font-family="monospace" fill="var(--gemma-text-secondary)">_proj</text>

              <circle cx="316" cy="120" r="9" fill="var(--gemma-surface)" stroke="var(--gemma-status-success)" stroke-width="1.4" />
              <text x="316" y="123.5" text-anchor="middle" font-size="10" font-weight="700" fill="var(--gemma-cat-tool-fg)">⊕</text>

              <!-- Sublayer B GeGLU particles rendered on top -->
              <circle r="3" fill="var(--gemma-status-warning)" data-anim-pts="32,120;90,120;104,82;234,82;244,120" data-dur="1880" data-phase="0" />
              <circle r="3" fill="var(--gemma-status-success)" data-anim-pts="32,120;90,120;104,159;234,159;244,120" data-dur="1880" data-phase="0" />
              <circle r="3.2" fill="var(--gemma-accent-fill)" data-anim-pts="254,120;316,120" data-dur="1320" data-phase="0.5" />
            </svg>
          `;
        }

        if (!bodyEl.querySelector('[data-role="attn-arch-stage"]')) {
          bodyEl.innerHTML = `
            <div class="llm-toolbar" data-role="attn-arch-toolbar" style="justify-content:space-between;margin-bottom:var(--gemma-space-3);">
              <div class="llm-toolbar__group" role="group" aria-label="Select model architecture">
                <span class="llm-toolbar__label">Model</span>
                ${GEMMA_MODELS_ARCH.map(
                  (mod, idx) =>
                    `<button type="button" class="llm-scenario-btn" data-mod="${idx}">${escapeHtml(mod.name)}</button>`
                ).join("")}
              </div>
              <button type="button" class="llm-action-btn" data-action="animate-layers"></button>
            </div>
            <div data-role="attn-arch-stage"></div>
          `;
        }

        bodyEl.querySelectorAll('[data-role="attn-arch-toolbar"] [data-mod]').forEach((btn) => {
          const isAct = Number(btn.dataset.mod) === modelIdx;
          btn.classList.toggle("llm-scenario-btn--active", isAct);
          btn.setAttribute("aria-pressed", String(isAct));
        });
        const animBtn = bodyEl.querySelector('[data-action="animate-layers"]');
        if (animBtn) {
          animBtn.classList.toggle("llm-action-btn--running", Boolean(cascadeTimer));
          animBtn.classList.toggle("llm-btn--primary", !cascadeTimer);
          const nextAnimLabel = cascadeTimer ? "⏸ Stop" : `▶ Forward Pass (L1 → L${m.layers})`;
          if (animBtn.textContent !== nextAnimLabel) animBtn.textContent = nextAnimLabel;
        }

        const archStageEl = bodyEl.querySelector('[data-role="attn-arch-stage"]');
        archStageEl.innerHTML = `
          <svg class="llm-svg-stage" viewBox="0 0 680 162" aria-label="${escapeHtml(m.fullName)} hybrid layer stack and KV sharing">
            <text x="16" y="20" font-size="9.4" font-weight="700" font-family="monospace" fill="var(--md-default-fg-color)">${escapeHtml(m.fullName)} · ${m.period - 1}:1 Local:Global</text>
            <g transform="translate(${m.doubleWideKvShared ? 298 : 356}, 10)" font-size="8" font-family="monospace">
              <rect x="0" y="1" width="10" height="10" rx="2" fill="var(--gemma-surface-sunken)" stroke="var(--gemma-border-strong)" />
              <text x="13" y="9" fill="var(--gemma-text-secondary)">Local ${m.windowShort}</text>
              <rect x="80" y="0" width="10" height="11" rx="2" fill="var(--gemma-accent-fill)" />
              <text x="93" y="9" fill="var(--gemma-text-secondary)">Global ${m.globalCtxShort}</text>
              <rect x="168" y="1" width="10" height="10" rx="2" fill="${hasKvShare ? "var(--gemma-cat-think-bg)" : "var(--gemma-cat-multi-bg)"}" stroke="${hasKvShare ? "var(--gemma-status-warning)" : "var(--gemma-cat-multi-border)"}" ${hasKvShare ? 'stroke-dasharray="2 1"' : ""} />
              <text x="181" y="9" fill="var(--gemma-text-secondary)">${hasKvShare ? `KV-Shared (${m.kvSharedLayers}L)` : `K=V (${m.globalGqa})`}</text>
              <rect x="270" y="1" width="10" height="10" rx="2" fill="${m.id === "26b" ? "var(--gemma-cat-multi-bg)" : "var(--gemma-cat-tool-bg)"}" stroke="${m.id === "26b" ? "var(--gemma-cat-multi-border)" : "var(--gemma-cat-tool-border)"}" />
              <text x="283" y="9" fill="var(--gemma-text-secondary)">${m.id === "26b" ? "MoE" : "FFN"}</text>
              ${
                m.doubleWideKvShared
                  ? `<rect x="310" y="1" width="10" height="10" rx="2" fill="var(--gemma-status-success)" fill-opacity="0.78" stroke="var(--gemma-cat-tool-border)" />
                     <text x="323" y="9" fill="var(--gemma-text-secondary)">2× FFN</text>`
                  : ""
              }
            </g>
            ${kvTopologySvg}
            ${layerColsSvg}
          </svg>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(17.5rem,1fr));gap:var(--gemma-space-3);margin-top:var(--gemma-space-3);">
            ${sublayerASvg}
            ${sublayerBSvg}
          </div>
          ${
            hasKvShare
              ? `<p style="margin:var(--gemma-space-3) 0 0;font-size:0.76rem;line-height:1.5;color:var(--gemma-text-muted);"><strong style="color:var(--gemma-text-secondary);">Note:</strong> <strong>E2B</strong> doubles FFN width (<code>6,144d → 12,288d</code>, highlighted in darker green) across its KV-shared layers (<code>L16–L35</code>) to reinvest saved K/V parameters, whereas <strong>E4B</strong> keeps a uniform <code>10,240d</code> FFN across all 42 layers.</p>`
              : ""
          }
        `;
      }

      cacheAnimNodes();
      stepParticles(typeof performance !== "undefined" ? performance.now() : Date.now());
      ensureRaf();
    };

    el.addEventListener("pointerdown", (e) => {
      if ((scanTimer || cascadeTimer) && e.target.closest("[data-word], [data-layer], [data-sent], [data-mod]")) {
        stopTimers();
      }
    });

    el.addEventListener("click", (e) => {
      if (e.target.closest('[data-action="auto-scan"]')) {
        if (scanTimer) {
          stopTimers();
          render();
          return;
        }
        stopTimers();
        wordIdx = 0;
        scanSubphase = 0;
        scanTimer = setInterval(() => {
          if (!el.isConnected) {
            stopTimers();
            return;
          }
          if (scanSubphase === 0) {
            scanSubphase = 1;
            render();
            return;
          }
          const s = ATTN_SENTENCES[sentIdx];
          if (wordIdx >= s.words.length - 1) {
            stopTimers();
            wordIdx = s.focusIdx;
            render();
            return;
          }
          wordIdx += 1;
          scanSubphase = 0;
          render();
        }, 540);
        render();
        return;
      }

      if (e.target.closest('[data-action="animate-layers"]')) {
        if (cascadeTimer) {
          stopTimers();
          render();
          return;
        }
        stopTimers();
        selectedLayer = 1;
        layerSubphase = 0;
        cascadeTimer = setInterval(() => {
          if (!el.isConnected) {
            stopTimers();
            return;
          }
          if (layerSubphase === 0) {
            layerSubphase = 1;
            render();
            return;
          }
          if (selectedLayer >= GEMMA_MODELS_ARCH[modelIdx].layers) {
            stopTimers();
            render();
            return;
          }
          selectedLayer += 1;
          layerSubphase = 0;
          render();
        }, 180);
        render();
        return;
      }

      const tabBtn = e.target.closest("[data-tab]");
      if (tabBtn) {
        stopTimers();
        tab = tabBtn.dataset.tab;
        render();
        return;
      }
      const sentBtn = e.target.closest("[data-sent]");
      if (sentBtn) {
        stopTimers();
        sentIdx = Number(sentBtn.dataset.sent);
        wordIdx = ATTN_SENTENCES[sentIdx].focusIdx;
        render();
        return;
      }
      const wordGroup = e.target.closest("[data-word]");
      if (wordGroup) {
        stopTimers();
        wordIdx = Number(wordGroup.dataset.word);
        render();
        return;
      }
      const modBtn = e.target.closest("[data-mod]");
      if (modBtn) {
        stopTimers();
        modelIdx = Number(modBtn.dataset.mod);
        selectedLayer = getDefaultLayerForModel(GEMMA_MODELS_ARCH[modelIdx]);
        render();
        return;
      }
      const layerBtn = e.target.closest("[data-layer]");
      if (layerBtn) {
        stopTimers();
        selectedLayer = Number(layerBtn.dataset.layer);
        render();
      }
    });

    render();
  };

  /* =====================================================================
     4. NEXT-TOKEN PREDICTION, SAMPLING SLIDERS & MTP (Step 4)
     ===================================================================== */
  const SAMPLING_PROMPTS = [
    {
      label: "Factual Geography",
      basePrompt: "The capital of France is",
      steps: [
        [
          { t: "▁Paris", id: 9079, logit: 14.2 },
          { t: "▁located", id: 5628, logit: 10.6 },
          { t: "▁known", id: 3224, logit: 9.8 },
          { t: "▁Lyon", id: 49533, logit: 8.4 },
          { t: "▁home", id: 2033, logit: 8.1 },
          { t: "▁a", id: 496, logit: 7.7 },
        ],
        [
          { t: ",", id: 236764, logit: 13.5 },
          { t: ".", id: 236761, logit: 13.1 },
          { t: "▁a", id: 496, logit: 10.9 },
          { t: "▁which", id: 837, logit: 10.4 },
          { t: "!", id: 236888, logit: 8.9 },
          { t: "▁and", id: 532, logit: 8.5 },
        ],
        [
          { t: "▁famous", id: 9639, logit: 12.8 },
          { t: "▁home", id: 2033, logit: 12.4 },
          { t: "▁the", id: 506, logit: 11.7 },
          { t: "▁known", id: 3224, logit: 11.2 },
          { t: "▁situated", id: 21988, logit: 9.6 },
          { t: "▁visited", id: 13896, logit: 8.8 },
        ],
        [
          { t: "▁for", id: 573, logit: 14.6 },
          { t: "▁throughout", id: 5459, logit: 10.8 },
          { t: "▁across", id: 3418, logit: 10.2 },
          { t: "▁around", id: 2101, logit: 9.7 },
          { t: "▁because", id: 1547, logit: 8.9 },
          { t: "▁in", id: 528, logit: 8.1 },
        ],
      ],
      mtpDraft: ["▁Paris", ",", "▁famous", "▁in"],
      mtpTarget: ["▁Paris", ",", "▁famous", "▁for"],
    },
    {
      label: "Creative Story",
      basePrompt: "Deep in the glowing forest, the little fox discovered a",
      steps: [
        [
          { t: "▁crystal", id: 12707, logit: 11.8 },
          { t: "▁hidden", id: 11497, logit: 11.6 },
          { t: "▁secret", id: 6789, logit: 11.2 },
          { t: "▁whispering", id: 146526, logit: 10.7 },
          { t: "▁golden", id: 13935, logit: 10.3 },
          { t: "▁spaceship", id: 100044, logit: 8.9 },
        ],
        [
          { t: "▁lantern", id: 41865, logit: 12.1 },
          { t: "▁portal", id: 22238, logit: 11.8 },
          { t: "▁key", id: 2307, logit: 11.4 },
          { t: "▁stream", id: 6381, logit: 10.8 },
          { t: "▁acorn", id: 217839, logit: 10.2 },
          { t: "▁map", id: 4187, logit: 9.6 },
        ],
        [
          { t: "▁that", id: 600, logit: 12.9 },
          { t: "▁glowing", id: 39622, logit: 11.7 },
          { t: "▁resting", id: 31432, logit: 11.1 },
          { t: "▁beneath", id: 25038, logit: 10.8 },
          { t: "▁floating", id: 18224, logit: 10.3 },
          { t: ".", id: 236761, logit: 9.4 },
        ],
        [
          { t: "▁was", id: 691, logit: 13.2 },
          { t: "▁could", id: 1451, logit: 12.5 },
          { t: "▁had", id: 1053, logit: 11.8 },
          { t: "▁would", id: 1093, logit: 11.1 },
          { t: "▁in", id: 528, logit: 10.4 },
          { t: "▁the", id: 506, logit: 9.6 },
        ],
      ],
      mtpDraft: ["▁crystal", "▁lantern", "▁that", "▁the"],
      mtpTarget: ["▁crystal", "▁lantern", "▁that", "▁was"],
    },
    {
      label: "Python Loop",
      basePrompt: "for i in range(",
      steps: [
        [
          { t: "1", id: 236770, logit: 13.4 },
          { t: "n", id: 236749, logit: 12.8 },
          { t: "len", id: 3469, logit: 12.2 },
          { t: "0", id: 236771, logit: 11.5 },
          { t: "5", id: 236810, logit: 10.4 },
          { t: "2", id: 236778, logit: 9.8 },
        ],
        [
          { t: "0", id: 236771, logit: 13.8 },
          { t: "):", id: 1473, logit: 13.2 },
          { t: ",", id: 236764, logit: 11.4 },
          { t: "4", id: 236812, logit: 9.8 },
          { t: ")", id: 236768, logit: 9.2 },
          { t: "6", id: 236825, logit: 8.5 },
        ],
        [
          { t: "):", id: 1473, logit: 14.9 },
          { t: ",", id: 236764, logit: 10.6 },
          { t: ")", id: 236768, logit: 9.9 },
          { t: "0", id: 236771, logit: 9.1 },
          { t: "]:", id: 9414, logit: 7.8 },
          { t: "▁", id: 236743, logit: 7.2 },
        ],
        [
          { t: "\\n", id: 107, logit: 14.8 },
          { t: "▁print", id: 1887, logit: 11.9 },
          { t: "▁pass", id: 1786, logit: 11.2 },
          { t: "▁return", id: 994, logit: 10.1 },
          { t: "▁if", id: 768, logit: 9.4 },
          { t: "▁▁▁▁", id: 140, logit: 8.7 },
        ],
      ],
      mtpDraft: ["1", "0", "):", "▁pass"],
      mtpTarget: ["1", "0", "):", "\\n"],
    },
  ];

  const getContextStepCandidates = (promptIdx, sIdx, generatedTokens) => {
    const p = SAMPLING_PROMPTS[promptIdx];
    const clampedIdx = Math.min(sIdx, p.steps.length - 1);
    if (clampedIdx === 0) return p.steps[0];
    const tok0 = generatedTokens[0];
    const tok1 = generatedTokens[1];
    const tok2 = generatedTokens[2];

    if (promptIdx === 0) {
      if (tok0 === "▁located") {
        if (clampedIdx === 1) {
          return [
            { t: "▁in", id: 528, logit: 14.3 },
            { t: "▁on", id: 580, logit: 11.8 },
            { t: "▁at", id: 657, logit: 10.5 },
            { t: "▁by", id: 684, logit: 9.7 },
            { t: "▁across", id: 3418, logit: 8.9 },
            { t: "▁around", id: 2101, logit: 8.2 },
          ];
        }
        if (clampedIdx === 2) {
          return [
            { t: "▁the", id: 506, logit: 13.7 },
            { t: "▁Paris", id: 9079, logit: 12.9 },
            { t: "▁France", id: 7001, logit: 11.8 },
            { t: "▁a", id: 496, logit: 10.6 },
            { t: "▁this", id: 672, logit: 9.4 },
            { t: "▁its", id: 1061, logit: 8.7 },
          ];
        }
        return [
          { t: ",", id: 236764, logit: 13.5 },
          { t: ".", id: 236761, logit: 13.1 },
          { t: "▁and", id: 532, logit: 11.4 },
          { t: "▁which", id: 837, logit: 10.6 },
          { t: "▁where", id: 1298, logit: 9.8 },
          { t: "!", id: 236888, logit: 8.5 },
        ];
      }
      if (tok0 === "▁known") {
        if (clampedIdx === 1) {
          return [
            { t: "▁as", id: 618, logit: 14.1 },
            { t: "▁for", id: 573, logit: 13.6 },
            { t: "▁to", id: 531, logit: 11.2 },
            { t: "▁in", id: 528, logit: 10.4 },
            { t: "▁throughout", id: 5459, logit: 9.6 },
            { t: "▁across", id: 3418, logit: 8.9 },
          ];
        }
        if (clampedIdx === 2) {
          return [
            { t: "▁Paris", id: 9079, logit: 13.9 },
            { t: "▁the", id: 506, logit: 12.8 },
            { t: "▁its", id: 1061, logit: 11.9 },
            { t: "▁a", id: 496, logit: 10.8 },
            { t: "▁being", id: 1646, logit: 9.7 },
            { t: "▁many", id: 1551, logit: 8.8 },
          ];
        }
        return [
          { t: ",", id: 236764, logit: 13.6 },
          { t: ".", id: 236761, logit: 13.2 },
          { t: "▁and", id: 532, logit: 11.5 },
          { t: "▁which", id: 837, logit: 10.7 },
          { t: "▁in", id: 528, logit: 9.6 },
          { t: "!", id: 236888, logit: 8.4 },
        ];
      }
      if (tok0 === "▁home") {
        if (clampedIdx === 1) {
          return [
            { t: "▁to", id: 531, logit: 14.5 },
            { t: "▁of", id: 529, logit: 12.1 },
            { t: "▁for", id: 573, logit: 10.8 },
            { t: "▁in", id: 528, logit: 9.6 },
            { t: "▁and", id: 532, logit: 8.9 },
            { t: ",", id: 236764, logit: 8.2 },
          ];
        }
        if (clampedIdx === 2) {
          return [
            { t: "▁many", id: 1551, logit: 13.6 },
            { t: "▁the", id: 506, logit: 12.9 },
            { t: "▁famous", id: 9639, logit: 12.1 },
            { t: "▁people", id: 1331, logit: 10.9 },
            { t: "▁more", id: 919, logit: 9.8 },
            { t: "▁some", id: 1070, logit: 8.9 },
          ];
        }
        return [
          { t: "▁people", id: 1331, logit: 13.4 },
          { t: "▁world", id: 1902, logit: 12.2 },
          { t: "▁of", id: 529, logit: 11.5 },
          { t: "▁in", id: 528, logit: 10.6 },
          { t: ",", id: 236764, logit: 9.7 },
          { t: ".", id: 236761, logit: 9.1 },
        ];
      }
      if (tok0 === "▁a") {
        if (clampedIdx === 1) {
          return [
            { t: "▁famous", id: 9639, logit: 13.7 },
            { t: "▁global", id: 4185, logit: 12.9 },
            { t: "▁large", id: 2455, logit: 11.8 },
            { t: "▁world", id: 1902, logit: 11.1 },
            { t: "▁very", id: 1401, logit: 9.8 },
            { t: "▁new", id: 861, logit: 8.6 },
          ];
        }
        if (clampedIdx === 2) {
          return [
            { t: "▁world", id: 1902, logit: 13.2 },
            { t: "▁home", id: 2033, logit: 12.4 },
            { t: "▁known", id: 3224, logit: 11.5 },
            { t: "▁Paris", id: 9079, logit: 10.8 },
            { t: "▁and", id: 532, logit: 9.6 },
            { t: ",", id: 236764, logit: 8.9 },
          ];
        }
        return p.steps[3];
      }

      if (clampedIdx === 2) {
        if (tok1 === "." || tok1 === "!") {
          return [
            { t: "▁It", id: 1030, logit: 13.4 },
            { t: "▁The", id: 669, logit: 12.6 },
            { t: "▁This", id: 1174, logit: 11.8 },
            { t: "▁Paris", id: 9079, logit: 11.1 },
            { t: "▁In", id: 799, logit: 9.9 },
            { t: "▁They", id: 2195, logit: 8.8 },
          ];
        }
        if (tok1 === "▁which" || tok1 === "▁and") {
          return [
            { t: "▁is", id: 563, logit: 13.8 },
            { t: "▁has", id: 815, logit: 12.6 },
            { t: "▁was", id: 691, logit: 11.7 },
            { t: "▁can", id: 740, logit: 10.5 },
            { t: "▁will", id: 795, logit: 9.6 },
            { t: "▁would", id: 1093, logit: 8.8 },
          ];
        }
      }
      if (clampedIdx === 3) {
        if (tok2 === "▁It" || tok2 === "▁This" || tok2 === "▁Paris") {
          return [
            { t: "▁is", id: 563, logit: 14.1 },
            { t: "▁has", id: 815, logit: 12.8 },
            { t: "▁was", id: 691, logit: 11.5 },
            { t: "▁can", id: 740, logit: 10.2 },
            { t: "▁will", id: 795, logit: 9.4 },
            { t: "▁must", id: 1921, logit: 8.6 },
          ];
        }
        if (tok2 === "▁The" || tok2 === "▁the") {
          return [
            { t: "▁most", id: 1346, logit: 13.9 },
            { t: "▁best", id: 1791, logit: 12.7 },
            { t: "▁world", id: 1902, logit: 11.8 },
            { t: "▁first", id: 1171, logit: 10.9 },
            { t: "▁large", id: 2455, logit: 9.6 },
            { t: "▁only", id: 1186, logit: 8.9 },
          ];
        }
        if (tok2 === "▁home") {
          return [
            { t: "▁to", id: 531, logit: 14.4 },
            { t: "▁of", id: 529, logit: 13.1 },
            { t: "▁for", id: 573, logit: 11.2 },
            { t: "▁in", id: 528, logit: 10.1 },
            { t: "▁and", id: 532, logit: 9.3 },
            { t: "▁with", id: 607, logit: 8.5 },
          ];
        }
        if (tok2 === "▁situated") {
          return [
            { t: "▁in", id: 528, logit: 14.3 },
            { t: "▁on", id: 580, logit: 12.9 },
            { t: "▁at", id: 657, logit: 11.1 },
            { t: "▁by", id: 684, logit: 10.4 },
            { t: "▁across", id: 3418, logit: 9.5 },
            { t: "▁around", id: 2101, logit: 8.8 },
          ];
        }
        if (tok2 === "▁visited") {
          return [
            { t: "▁by", id: 684, logit: 14.5 },
            { t: "▁in", id: 528, logit: 11.8 },
            { t: "▁throughout", id: 5459, logit: 10.9 },
            { t: "▁around", id: 2101, logit: 9.8 },
            { t: "▁for", id: 573, logit: 9.1 },
            { t: "▁because", id: 1547, logit: 8.4 },
          ];
        }
        if (tok1 === "▁which" || tok1 === "▁and") {
          return [
            { t: "▁famous", id: 9639, logit: 13.6 },
            { t: "▁known", id: 3224, logit: 12.8 },
            { t: "▁home", id: 2033, logit: 11.9 },
            { t: "▁located", id: 5628, logit: 11.1 },
            { t: "▁situated", id: 21988, logit: 9.7 },
            { t: "▁visited", id: 13896, logit: 8.9 },
          ];
        }
      }
    }

    if (promptIdx === 2) {
      if (tok0 === "len" && clampedIdx === 1) {
        return [
          { t: "(", id: 236769, logit: 14.4 },
          { t: "):", id: 1473, logit: 10.8 },
          { t: ",", id: 236764, logit: 9.6 },
          { t: ")", id: 236768, logit: 9.1 },
          { t: "▁", id: 236743, logit: 8.2 },
          { t: "0", id: 236771, logit: 7.5 },
        ];
      }
      if (tok1 === "):" && clampedIdx === 2) {
        return p.steps[3];
      }
      if (tok2 === "\\n" && clampedIdx === 3) {
        return [
          { t: "▁▁▁▁", id: 140, logit: 14.5 },
          { t: "▁print", id: 1887, logit: 12.4 },
          { t: "▁pass", id: 1786, logit: 11.3 },
          { t: "▁return", id: 994, logit: 10.2 },
          { t: "▁if", id: 768, logit: 9.5 },
          { t: "▁for", id: 573, logit: 8.8 },
        ];
      }
    }

    return p.steps[clampedIdx];
  };

  const mountSampling = (el) => {
    let promptIdx = 0;
    let stepIdx = 0;
    let generatedTokens = [];
    let temp = 1.0;
    let topK = 64;
    let topP = 0.95;
    let mtpMode = false;
    // MTP speculative step: "idle" → "drafting" → "verifying" → "accepting" → "rejecting" → "done"
    let mtpPhase = "idle";
    let mtpDraftedCount = 0;
    let mtpVerifiedCount = 0;
    // One-shot animation targets (cleared right after render so re-renders don't replay them).
    let mtpEnterIdx = -1;
    let mtpPopIdx = -1;
    let justSampledIdx = -1;
    let mtpTimer = null;
    let autoGenTimer = null;
    let autoGenRunning = false;
    let sweepTimer = null;
    let scanningRowIdx = -1;

    el.innerHTML = `
      <div class="llm-widget__header">
        <p class="llm-widget__title">Next-Token Sampling &amp; Speculative Decoding</p>
        <div class="llm-toolbar__group" role="group" aria-label="Decoding mode">
          <span class="llm-toolbar__label">Mode</span>
          <div class="llm-seg">
            <button type="button" class="llm-seg__btn llm-seg__btn--active" data-mtp-mode="off">Standard (1-Token)</button>
            <button type="button" class="llm-seg__btn" data-mtp-mode="on">Gemma 4 MTP (4-Token)</button>
          </div>
        </div>
      </div>
      <div class="llm-toolbar" style="justify-content:space-between;margin-bottom:var(--gemma-space-3);">
        <div class="llm-toolbar__group" role="group" aria-label="Select prompt scenario" data-role="prompts"></div>
        <div class="llm-toolbar__group" role="group" aria-label="Sampling actions">
          <button type="button" class="llm-action-btn llm-btn--primary" data-action="sample">🎲 Sample Next</button>
          <button type="button" class="llm-action-btn" data-action="auto-gen">▶ Auto-Generate</button>
          <button type="button" class="llm-action-btn llm-action-btn--ghost" data-action="reset-prompt">↺ Reset</button>
        </div>
      </div>
      <div class="llm-sampling-stage">
        <div class="llm-panel llm-sampling-stage__seq" data-role="sequence"></div>
        <div class="llm-panel llm-sampling-stage__bars" data-role="bars">
          <div data-role="bars-head"></div>
          <div data-role="bars-list"></div>
        </div>
      </div>
      <div class="llm-sampling-params" aria-label="Sampling parameters">
        <div class="llm-sampling-params__head">
          <span class="llm-toolbar__label">Sampling Parameters</span>
          <button type="button" class="llm-action-btn llm-action-btn--ghost" data-action="reset-defaults" style="padding:0.15rem 0.5rem;font-size:0.7rem;">Reset Defaults (T=1.0, K=64, P=0.95)</button>
        </div>
        <div class="llm-slider-grid" style="margin-bottom:0;">
          <label class="llm-slider-card">
            <div style="display:flex;justify-content:space-between;">
              <strong>Temperature (T)</strong><code data-role="temp-val">1.00</code>
            </div>
            <input type="range" min="0" max="2" step="0.05" value="1.0" data-role="temp" />
          </label>
          <label class="llm-slider-card">
            <div style="display:flex;justify-content:space-between;">
              <strong>Top-K</strong><code data-role="topk-val">64</code>
            </div>
            <input type="range" min="1" max="64" step="1" value="64" data-role="topk" />
          </label>
          <label class="llm-slider-card">
            <div style="display:flex;justify-content:space-between;">
              <strong>Top-P (Nucleus)</strong><code data-role="topp-val">0.95</code>
            </div>
            <input type="range" min="0.10" max="1.00" step="0.05" value="0.95" data-role="topp" />
          </label>
        </div>
      </div>
    `;

    const promptsEl = el.querySelector('[data-role="prompts"]');
    const seqEl = el.querySelector('[data-role="sequence"]');
    const barsEl = el.querySelector('[data-role="bars"]');
    const barsHeadEl = barsEl.querySelector('[data-role="bars-head"]');
    const barsListEl = barsEl.querySelector('[data-role="bars-list"]');
    const tempInput = el.querySelector('[data-role="temp"]');
    const topkInput = el.querySelector('[data-role="topk"]');
    const toppInput = el.querySelector('[data-role="topp"]');
    const barRows = [];
    const cutoffEl = document.createElement("div");
    cutoffEl.className = "llm-cutoff-divider";

    const createBarRow = () => {
      const row = document.createElement("div");
      row.className = "llm-bar-row";
      row.innerHTML = `
        <div>
          <code data-role="tok"></code>
          <span data-role="tok-id" style="font-size:0.68rem;color:var(--gemma-text-muted)"></span>
        </div>
        <div class="llm-bar-track">
          <div class="llm-bar-fill" style="width:2%"></div>
        </div>
        <div data-role="pct" style="text-align:right;font-family:monospace;font-weight:700;"></div>
        <div data-role="status" style="text-align:right;font-size:0.7rem;"></div>
      `;
      return {
        row,
        tokEl: row.querySelector('[data-role="tok"]'),
        idEl: row.querySelector('[data-role="tok-id"]'),
        fillEl: row.querySelector(".llm-bar-fill"),
        pctEl: row.querySelector('[data-role="pct"]'),
        statusEl: row.querySelector('[data-role="status"]'),
      };
    };

    const clearTimers = () => {
      autoGenRunning = false;
      if (autoGenTimer) {
        clearTimeout(autoGenTimer);
        autoGenTimer = null;
      }
      if (sweepTimer) {
        clearInterval(sweepTimer);
        sweepTimer = null;
      }
      if (mtpTimer) {
        clearTimeout(mtpTimer);
        mtpTimer = null;
        if (mtpPhase === "accepting" || mtpPhase === "rejecting") {
          const p = SAMPLING_PROMPTS[promptIdx];
          generatedTokens = [...p.mtpTarget];
          mtpDraftedCount = p.mtpDraft.length;
          mtpVerifiedCount = p.mtpDraft.length;
          mtpPhase = "done";
        } else if (mtpPhase !== "done") {
          // Interrupted mid-step before acceptance: drop unverified drafts.
          resetMtp();
        }
      }
      scanningRowIdx = -1;
    };

    const resetMtp = () => {
      mtpPhase = "idle";
      mtpDraftedCount = 0;
      mtpVerifiedCount = 0;
    };

    const fmtTok = (tok) => (tok === "\\n" ? "↵" : tok === "▁▁▁▁" ? "····" : tok.replace(/^▁/, " "));

    const computeDistribution = (targetStepIdx = stepIdx) => {
      const rawCandidates = getContextStepCandidates(promptIdx, targetStepIdx, generatedTokens);
      const capped = rawCandidates.map((c) => ({
        ...c,
        softLogit: 30.0 * Math.tanh(c.logit / 30.0),
      }));
      const sorted = [...capped].sort((a, b) => b.softLogit - a.softLogit);

      const effTemp = Math.max(0.02, temp);
      const maxL = sorted[0].softLogit;
      const exps = sorted.map((c) => (temp === 0 ? (c === sorted[0] ? 1 : 0) : Math.exp((c.softLogit - maxL) / effTemp)));
      const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
      let cumP = 0;

      return sorted.map((c, idx) => {
        const prob = exps[idx] / sumExp;
        const prevCum = cumP;
        cumP += prob;
        const keptByK = idx < topK;
        const keptByP = idx === 0 || prevCum < topP;
        const active = keptByK && keptByP;
        let reason = "Active (In Nucleus)";
        if (!keptByK) reason = `Filtered (Top-K > ${topK})`;
        else if (!keptByP) reason = `Filtered (Top-P > ${topP.toFixed(2)})`;
        return { ...c, prob, cumP, active, reason };
      });
    };

    const render = () => {
      el.querySelector('[data-role="temp-val"]').textContent = temp.toFixed(2);
      el.querySelector('[data-role="topk-val"]').textContent = String(topK);
      el.querySelector('[data-role="topp-val"]').textContent = topP.toFixed(2);
      el.querySelectorAll("[data-mtp-mode]").forEach((btn) => {
        const isActive = (btn.dataset.mtpMode === "on") === mtpMode;
        btn.classList.toggle("llm-seg__btn--active", isActive);
        btn.setAttribute("aria-pressed", String(isActive));
      });

      const autoBtn = el.querySelector('[data-action="auto-gen"]');
      autoBtn.classList.toggle("llm-action-btn--running", autoGenRunning);
      autoBtn.textContent = autoGenRunning ? "⏸ Stop Auto-Generate" : "▶ Auto-Generate";

      if (!promptsEl.children.length) {
        promptsEl.innerHTML = SAMPLING_PROMPTS.map(
          (p, i) => `<button type="button" class="llm-scenario-btn" data-prompt="${i}">${escapeHtml(p.label)}</button>`
        ).join("");
      }
      promptsEl.querySelectorAll("[data-prompt]").forEach((btn) => {
        const isAct = Number(btn.dataset.prompt) === promptIdx;
        btn.classList.toggle("llm-scenario-btn--active", isAct);
        btn.setAttribute("aria-pressed", String(isAct));
      });

      const p = SAMPLING_PROMPTS[promptIdx];
      const totalSteps = p.steps.length;
      const clampedStepIdx = Math.min(stepIdx, totalSteps - 1);
      const sampledAtStep = generatedTokens[clampedStepIdx];
      const streamStepLabel =
        generatedTokens.length === 0
          ? `Step 1 of ${totalSteps} — Ready to sample`
          : `Step ${clampedStepIdx + 1} of ${totalSteps} (${generatedTokens.length}/${totalSteps} sampled)`;

      const mtpRunning =
        mtpMode &&
        (mtpPhase === "drafting" || mtpPhase === "verifying" || mtpPhase === "accepting" || mtpPhase === "rejecting");
      const ghostHtml = mtpRunning
        ? p.mtpDraft
          .slice(mtpVerifiedCount, mtpDraftedCount)
          .map((tok, i) => {
            const idx = mtpVerifiedCount + i;
            const isRejecting = mtpPhase === "rejecting" && idx === mtpVerifiedCount;
            const cls = [
              "llm-token-ghost",
              idx === mtpEnterIdx ? "llm-token-ghost--enter" : "",
              mtpPhase === "verifying" ? "llm-token-ghost--checking" : "",
              isRejecting ? "llm-token-ghost--rejected" : "",
            ].join(" ");
            return `<span class="${cls}" title="${isRejecting ? "Draft missed — target model replacing" : "Draft token (not yet verified)"}">${escapeHtml(fmtTok(tok))}</span>`;
          })
          .join("")
        : "";

      seqEl.innerHTML = `
        <div style="font-size:0.78rem;color:var(--gemma-text-muted);margin-bottom:0.35rem;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.35rem;">
          <span><strong>Generated Sequence</strong> · ${streamStepLabel}</span>
          ${mtpMode ? `<span class="llm-widget__badge">⚡ 3.2× Speculative Speedup</span>` : ""}
        </div>
        <div style="font-family:var(--md-code-font-family, monospace);font-size:0.95rem;display:flex;flex-wrap:wrap;align-items:center;gap:3px;min-height:1.9rem;">
          <span>${escapeHtml(p.basePrompt)}</span>${generatedTokens
          .map((tok, idx) => {
            const isInspected = idx === clampedStepIdx;
            const displayTok = fmtTok(tok);
            return `<button type="button" class="token-turn ${idx === justSampledIdx ? "llm-token-fly" : ""}" data-step-idx="${idx}" title="Click to inspect Step ${idx + 1} distribution (${escapeHtml(tok)})" style="margin-left:2px;cursor:pointer;border:${isInspected ? "2px solid var(--gemma-accent)" : "1px solid transparent"};font-family:inherit;font-size:inherit;">${escapeHtml(displayTok)}</button>`;
          })
          .join("")}${ghostHtml}${!mtpRunning && generatedTokens.length < totalSteps && clampedStepIdx === generatedTokens.length
            ? `<span class="llm-widget__badge" style="margin-left:4px;opacity:0.85;">${scanningRowIdx >= 0 ? "🎲 Rolling…" : "Next: Step " + (generatedTokens.length + 1)}</span>`
            : ""
          }
        </div>
        ${mtpMode ? renderMtpPanel(p) : ""}
      `;

      const dist = computeDistribution(clampedStepIdx);
      const stepPillsHtml = p.steps
        .map((_, idx) => {
          const isCurrent = idx === clampedStepIdx;
          const tok = generatedTokens[idx];
          const canSelect = idx <= generatedTokens.length;
          const wasReplaced = mtpMode && tok && p.mtpDraft[idx] !== p.mtpTarget[idx] && tok === p.mtpTarget[idx];
          const label = tok
            ? `Step ${idx + 1}: ${tok.replace(/^▁/, "")} ${wasReplaced ? "↻" : "✓"}`
            : idx === generatedTokens.length
              ? `Step ${idx + 1} (Next)`
              : `Step ${idx + 1}`;
          return `<button type="button" class="llm-seg__btn ${isCurrent ? "llm-seg__btn--active" : ""}" data-step-idx="${idx}" ${canSelect ? "" : "disabled"}>${escapeHtml(label)}</button>`;
        })
        .join("");

      barsHeadEl.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;margin-bottom:0.45rem;font-size:0.78rem;color:var(--gemma-text-muted);">
          <span><strong style="color:var(--md-default-fg-color);">Candidate Probabilities</strong> · ${sampledAtStep !== undefined
          ? `Sampled <code>${escapeHtml(sampledAtStep)}</code>`
          : "Ready to sample"
        }</span>
          <div class="llm-seg">${stepPillsHtml}</div>
        </div>`;

      const stepDraft = p.mtpDraft[clampedStepIdx];
      const stepTarget = p.mtpTarget[clampedStepIdx];
      const isMtpMissStep =
        mtpMode &&
        stepDraft !== stepTarget &&
        (sampledAtStep !== undefined || (mtpPhase === "rejecting" && clampedStepIdx === mtpVerifiedCount));

      while (barRows.length < dist.length) {
        const entry = createBarRow();
        barRows.push(entry);
        barsListEl.appendChild(entry.row);
      }
      while (barRows.length > dist.length) {
        barRows.pop().row.remove();
      }

      let firstFilteredIdx = -1;
      dist.forEach((item, idx) => {
        if (!item.active && firstFilteredIdx === -1) firstFilteredIdx = idx;
        const pct = (item.prob * 100).toFixed(1);
        const isScan = idx === scanningRowIdx;
        const isWin = scanningRowIdx === -1 && sampledAtStep !== undefined && item.t === sampledAtStep;
        const isDraftMissRow = isMtpMissStep && item.t === stepDraft;
        const statusLabel = isWin
          ? isMtpMissStep
            ? "🎯 Target Replaced!"
            : "🎲 Sampled!"
          : isDraftMissRow
            ? `✗ Drafter Miss (${item.active ? "Low Prob" : item.reason})`
            : item.active
              ? "✓ Kept"
              : item.reason;
        const statusColor = isDraftMissRow
          ? "var(--gemma-cat-role-fg)"
          : isWin || item.active
            ? "var(--gemma-accent)"
            : "var(--gemma-text-muted)";

        const entry = barRows[idx];
        entry.row.classList.toggle("llm-bar-row--filtered", !item.active);
        entry.row.classList.toggle("llm-bar-row--scanning", isScan);
        entry.row.classList.toggle("llm-bar-row--winner", isWin);
        entry.tokEl.textContent = item.t;
        entry.idEl.textContent = `#${item.id}`;
        entry.fillEl.style.width = `${Math.max(2, item.prob * 100)}%`;
        entry.pctEl.textContent = `${pct}%`;
        entry.statusEl.textContent = statusLabel;
        entry.statusEl.style.color = statusColor;
      });

      if (firstFilteredIdx >= 0) {
        cutoffEl.textContent = `✂ Nucleus / Top-K Cutoff (Top-P = ${topP.toFixed(2)}, Top-K = ${topK}) — Tokens below are zeroed out`;
        const anchorRow = barRows[firstFilteredIdx].row;
        if (cutoffEl.parentNode !== barsListEl || cutoffEl.nextElementSibling !== anchorRow) {
          barsListEl.insertBefore(cutoffEl, anchorRow);
        }
      } else if (cutoffEl.parentNode) {
        cutoffEl.remove();
      }

      mtpEnterIdx = -1;
      mtpPopIdx = -1;
      justSampledIdx = -1;
    };

    const renderMtpPanel = (p) => {
      const n = p.mtpDraft.length;
      const phaseOrder = { idle: -1, drafting: 0, verifying: 1, accepting: 2, rejecting: 2, done: 3 };
      const cur = phaseOrder[mtpPhase];
      const phases = [`① Draft ×${n}`, "② Verify (1 target pass)", "③ Accept & Replace"];
      const phaseHtml = phases
        .map((label, i) => {
          const state = cur === i ? "llm-mtp-phase--active" : cur > i ? "llm-mtp-phase--done" : "";
          return `<span class="llm-mtp-phase ${state}">${cur > i ? "✓ " : ""}${escapeHtml(label)}</span>`;
        })
        .join(`<span class="llm-mtp-arrow">→</span>`);

      const chipsHtml = p.mtpDraft
        .map((d, idx) => {
          const targetTok = p.mtpTarget[idx];
          const isMiss = d !== targetTok;
          let cls;
          let innerHtml;
          if (idx < mtpVerifiedCount) {
            if (isMiss) {
              cls = `llm-mtp-chip--replaced ${idx === mtpPopIdx ? "llm-mtp-chip--accept" : ""}`;
              innerHtml = `<s class="llm-mtp-chip__miss" title="Drafter proposed ${escapeHtml(d)} (rejected by target model)">${escapeHtml(d)}</s> <span class="llm-mtp-chip__arrow">→</span> ${escapeHtml(targetTok)} <span class="llm-mtp-chip__icon">↻</span>`;
            } else {
              cls = `llm-mtp-chip--verified ${idx === mtpPopIdx ? "llm-mtp-chip--accept" : ""}`;
              innerHtml = `${escapeHtml(d)} <span class="llm-mtp-chip__icon">✓</span>`;
            }
          } else if (mtpPhase === "rejecting" && idx === mtpVerifiedCount) {
            cls = "llm-mtp-chip--rejected";
            innerHtml = `<s class="llm-mtp-chip__miss">${escapeHtml(d)}</s> <span class="llm-mtp-chip__icon">✗</span>`;
          } else if (idx < mtpDraftedCount) {
            cls = `llm-mtp-chip--draft ${idx === mtpEnterIdx ? "llm-mtp-chip--enter" : ""} ${mtpPhase === "verifying" ? "llm-mtp-chip--checking" : ""}`;
            const icon = mtpPhase === "verifying" ? "⟳" : "…";
            innerHtml = `${escapeHtml(d)} <span class="llm-mtp-chip__icon">${icon}</span>`;
          } else {
            cls = "llm-mtp-chip--slot";
            innerHtml = `t+${idx + 1}`;
          }
          return `<span class="llm-mtp-chip ${cls}">${innerHtml}</span>`;
        })
        .join("");

      const acceptedDrafts = p.mtpDraft.filter((d, i) => d === p.mtpTarget[i]).length;
      const missIdx = p.mtpDraft.findIndex((d, i) => d !== p.mtpTarget[i]);
      let status;
      if (mtpPhase === "drafting") status = `Drafter proposing token ${mtpDraftedCount}/${n}…`;
      else if (mtpPhase === "verifying") status = `Target model scoring all ${n} drafts in one parallel forward pass…`;
      else if (mtpPhase === "accepting") status = `Accepted ${mtpVerifiedCount}/${n} — committing verified drafts to context…`;
      else if (mtpPhase === "rejecting")
        status = `✗ Draft #${missIdx + 1} (${p.mtpDraft[missIdx]}) missed — replacing with target model's prediction (${p.mtpTarget[missIdx]})…`;
      else if (mtpPhase === "done")
        status = `✓ ${acceptedDrafts}/${n} drafts accepted · Final draft (${p.mtpDraft[missIdx]}) missed → replaced with target model's ${p.mtpTarget[missIdx]} in 1 pass`;
      else status = "Press “Sample Next” to run one speculative step";

      const drafterBlocks = p.mtpDraft
        .map((_, i) => `<span class="llm-mtp-cost__block llm-mtp-cost__block--mini ${i < mtpDraftedCount ? "llm-mtp-cost__block--lit" : ""}"></span>`)
        .join("");
      const standardBlocks = p.mtpDraft.map(() => `<span class="llm-mtp-cost__block llm-mtp-cost__block--muted"></span>`).join("");
      const targetCls = `${cur >= 1 ? "llm-mtp-cost__block--lit" : ""} ${mtpPhase === "verifying" ? "llm-mtp-cost__block--busy" : ""}`;

      return `<div class="llm-mtp-panel">
          <div class="llm-mtp-phases">${phaseHtml}</div>
          <div class="llm-mtp-row">
            <span class="token-think">MTP Drafter (4-Layer Assistant)</span>
            <div class="llm-mtp-track ${mtpPhase === "verifying" ? "llm-mtp-track--verifying" : ""} ${mtpPhase === "done" ? "llm-mtp-track--done" : ""}">${chipsHtml}</div>
          </div>
          <div class="llm-mtp-status ${mtpPhase === "done" ? "llm-mtp-status--done" : ""}" aria-live="polite">${escapeHtml(status)}</div>
          <div class="llm-mtp-cost">
            <span>Standard</span>
            <span class="llm-mtp-cost__blocks">${standardBlocks}</span>
            <span>${n} target passes</span>
            <span>MTP</span>
            <span class="llm-mtp-cost__blocks">${drafterBlocks}<span class="llm-mtp-cost__plus">+</span><span class="llm-mtp-cost__block ${targetCls}"></span></span>
            <span>${n} light drafter steps + 1 target pass</span>
          </div>
        </div>`;
    };

    const runMtpAnimation = () => {
      const p = SAMPLING_PROMPTS[promptIdx];
      const n = p.mtpDraft.length;
      if (mtpTimer) {
        clearTimeout(mtpTimer);
        mtpTimer = null;
      }
      scanningRowIdx = -1;

      generatedTokens = [];
      stepIdx = 0;
      mtpDraftedCount = 0;
      mtpVerifiedCount = 0;
      mtpPhase = "drafting";
      render();

      const schedule = (fn, ms) => {
        mtpTimer = setTimeout(() => {
          mtpTimer = null;
          if (!el.isConnected) return;
          fn();
        }, ms);
      };

      const replaceFinal = () => {
        const idx = mtpVerifiedCount;
        generatedTokens.push(p.mtpTarget[idx]);
        mtpPopIdx = idx;
        justSampledIdx = idx;
        mtpVerifiedCount += 1;
        stepIdx = Math.min(mtpVerifiedCount - 1, p.steps.length - 1);
        mtpPhase = "done";
        render();
      };

      // ③ Accept & Replace: verified drafts are committed; a missed draft is replaced by the target model's prediction.
      const acceptTick = () => {
        const idx = mtpVerifiedCount;
        if (p.mtpDraft[idx] !== p.mtpTarget[idx]) {
          mtpPhase = "rejecting";
          stepIdx = Math.min(idx, p.steps.length - 1);
          render();
          schedule(replaceFinal, 560);
          return;
        }
        mtpPhase = "accepting";
        generatedTokens.push(p.mtpTarget[idx]);
        mtpPopIdx = idx;
        justSampledIdx = idx;
        mtpVerifiedCount += 1;
        stepIdx = Math.min(mtpVerifiedCount - 1, p.steps.length - 1);
        if (mtpVerifiedCount >= n) {
          mtpPhase = "done";
          render();
          return;
        }
        render();
        schedule(acceptTick, 280);
      };

      // ② Verify: the target model checks every draft position in a single pass.
      const verify = () => {
        mtpPhase = "verifying";
        render();
        schedule(acceptTick, 950);
      };

      // ① Draft: the lightweight drafter proposes tokens autoregressively.
      const draftTick = () => {
        mtpEnterIdx = mtpDraftedCount;
        mtpDraftedCount += 1;
        render();
        if (mtpDraftedCount < n) schedule(draftTick, 260);
        else schedule(verify, 380);
      };

      schedule(draftTick, 200);
    };

    const prepareNextSampleStep = () => {
      const p = SAMPLING_PROMPTS[promptIdx];
      if (generatedTokens.length >= p.steps.length) {
        generatedTokens = [];
        stepIdx = 0;
      } else {
        stepIdx = generatedTokens.length;
      }
    };

    const commitSample = () => {
      const dist = computeDistribution(stepIdx);
      const activeList = dist.filter((d) => d.active);
      const total = activeList.reduce((acc, d) => acc + d.prob, 0) || 1;
      let r = Math.random() * total;
      let chosen = activeList[0] || dist[0];
      for (const cand of activeList) {
        r -= cand.prob;
        if (r <= 0) {
          chosen = cand;
          break;
        }
      }
      scanningRowIdx = -1;
      generatedTokens[stepIdx] = chosen.t;
      justSampledIdx = stepIdx;
      render();
    };

    const sampleNext = (withSweep = false, onComplete = null) => {
      if (mtpMode) {
        runMtpAnimation();
        return;
      }
      if (sweepTimer) {
        clearInterval(sweepTimer);
        sweepTimer = null;
        commitSample();
      }
      prepareNextSampleStep();
      if (!withSweep) {
        commitSample();
        if (onComplete) onComplete();
        return;
      }
      const dist = computeDistribution(stepIdx);
      const activeCount = Math.max(1, dist.filter((d) => d.active).length);
      const totalTicks = activeCount === 1 ? 4 : Math.max(6, Math.min(10, activeCount + 3));
      let ticks = 0;
      const advanceSweep = () => {
        if (ticks >= totalTicks) {
          if (sweepTimer) {
            clearInterval(sweepTimer);
            sweepTimer = null;
          }
          commitSample();
          if (onComplete) onComplete();
          return;
        }
        scanningRowIdx = activeCount === 1 ? (ticks % 2 === 0 ? 0 : -1) : ticks % activeCount;
        ticks += 1;
        render();
      };
      advanceSweep();
      sweepTimer = setInterval(() => {
        if (!el.isConnected) {
          clearTimers();
          return;
        }
        advanceSweep();
      }, 75);
    };

    tempInput.addEventListener("input", () => {
      temp = Number(tempInput.value);
      render();
    });
    topkInput.addEventListener("input", () => {
      topK = Number(topkInput.value);
      render();
    });
    toppInput.addEventListener("input", () => {
      topP = Number(toppInput.value);
      render();
    });

    el.addEventListener("click", (e) => {
      const pBtn = e.target.closest("[data-prompt]");
      if (pBtn) {
        clearTimers();
        promptIdx = Number(pBtn.dataset.prompt);
        stepIdx = 0;
        generatedTokens = [];
        resetMtp();
        render();
        return;
      }
      const stepBtn = e.target.closest("[data-step-idx]");
      if (stepBtn && !stepBtn.disabled) {
        clearTimers();
        const targetIdx = Number(stepBtn.dataset.stepIdx);
        const p = SAMPLING_PROMPTS[promptIdx];
        if (targetIdx >= 0 && targetIdx <= Math.min(generatedTokens.length, p.steps.length - 1)) {
          stepIdx = targetIdx;
          render();
        }
        return;
      }
      const mtpSegBtn = e.target.closest("[data-mtp-mode]");
      if (mtpSegBtn) {
        const nextMtp = mtpSegBtn.dataset.mtpMode === "on";
        if (nextMtp !== mtpMode) {
          clearTimers();
          mtpMode = nextMtp;
          if (mtpMode) runMtpAnimation();
          else render();
        }
        return;
      }
      const act = e.target.closest("[data-action]")?.dataset.action;
      if (act === "sample") {
        autoGenRunning = false;
        if (autoGenTimer) {
          clearTimeout(autoGenTimer);
          autoGenTimer = null;
        }
        sampleNext(true);
      } else if (act === "auto-gen") {
        if (autoGenRunning) {
          clearTimers();
          render();
          return;
        }
        clearTimers();
        if (mtpMode) {
          runMtpAnimation();
          return;
        }
        stepIdx = 0;
        generatedTokens = [];
        autoGenRunning = true;
        const runAutoStep = () => {
          if (!autoGenRunning || !el.isConnected) {
            clearTimers();
            return;
          }
          const p = SAMPLING_PROMPTS[promptIdx];
          if (generatedTokens.length >= p.steps.length) {
            autoGenRunning = false;
            render();
            return;
          }
          sampleNext(true, () => {
            if (!autoGenRunning || !el.isConnected) return;
            const curP = SAMPLING_PROMPTS[promptIdx];
            if (generatedTokens.length >= curP.steps.length) {
              autoGenRunning = false;
              render();
              return;
            }
            autoGenTimer = setTimeout(() => {
              autoGenTimer = null;
              runAutoStep();
            }, 380);
          });
        };
        runAutoStep();
      } else if (act === "reset-prompt") {
        clearTimers();
        stepIdx = 0;
        generatedTokens = [];
        resetMtp();
        render();
      } else if (act === "reset-defaults") {
        temp = 1.0;
        topK = 64;
        topP = 0.95;
        tempInput.value = "1.0";
        topkInput.value = "64";
        toppInput.value = "0.95";
        render();
      } else if (act === "toggle-mtp") {
        clearTimers();
        mtpMode = !mtpMode;
        if (mtpMode) runMtpAnimation();
        else render();
      }
    });

    render();
  };

  /* =====================================================================
     5. END-TO-END PLAYGROUND: CHAT, THINKING (<|think|>) & TOOLS (Step 5)
     ===================================================================== */
  const PLAYGROUND_SYSTEM_PROMPT = "You are a helpful assistant.";

  const PLAYGROUND_CAT_THUMB_SVG = `<svg width="60" height="44" viewBox="0 0 60 44" aria-label="Orange tabby cat sleeping on a sunny windowsill" role="img" style="display:block;border-radius:4px;flex-shrink:0;border:1px solid var(--gemma-border);">
    <rect width="60" height="34" fill="#dbeafe"/>
    <polygon points="8,0 38,0 54,34 18,34" fill="#fef08a" opacity="0.65"/>
    <line x1="30" y1="0" x2="30" y2="34" stroke="#cbd5e1" stroke-width="1.2"/>
    <line x1="0" y1="17" x2="60" y2="17" stroke="#cbd5e1" stroke-width="1.2"/>
    <rect y="34" width="60" height="10" fill="#d97706"/>
    <rect y="34" width="60" height="2" fill="#f59e0b"/>
    <ellipse cx="33" cy="31" rx="14" ry="8.5" fill="#f97316"/>
    <path d="M29,23.5 Q31,27.5 29.5,31 M35,23 Q37,27.5 35.5,31.5 M40.5,24.5 Q42,28 40.5,31.5" stroke="#c2410c" stroke-width="1.4" stroke-linecap="round" fill="none"/>
    <path d="M44,33 Q52,33 50,27.5 Q48.5,25 45.5,27.5" stroke="#fb923c" stroke-width="3.2" stroke-linecap="round" fill="none"/>
    <polygon points="16.5,24 18,16.5 23,21.5" fill="#ea580c"/>
    <polygon points="18,23 18.8,18.2 22,21.5" fill="#fed7aa"/>
    <polygon points="24.5,21.5 29,16.5 30.5,24" fill="#ea580c"/>
    <polygon points="25.5,21.5 28.4,18.2 29.3,23" fill="#fed7aa"/>
    <circle cx="23.5" cy="27.5" r="7" fill="#fb923c"/>
    <path d="M19.5,27.2 Q21.2,29 22.8,27.2 M24.5,27.2 Q26.2,29 27.8,27.2" stroke="#7c2d12" stroke-width="1.2" stroke-linecap="round" fill="none"/>
    <polygon points="22.7,29.5 24.5,29.5 23.6,30.7" fill="#f43f5e"/>
    <text x="45" y="13" font-size="6.5" font-family="monospace" font-weight="700" fill="#475569">zZ</text>
  </svg>`;

  const PLAYGROUND_SCENARIOS = [
    {
      id: "math",
      label: "Math & Logic Puzzle",
      userPrompt: "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much is the ball?",
      thought: "Let ball = x. Then bat = x + 1.00. Total: x + (x + 1.00) = 1.10 → 2x = 0.10 → x = $0.05 (5 cents).",
      answer: "The ball costs <strong>$0.05 (5 cents)</strong>, and the bat costs <strong>$1.05</strong> ($0.05 + $1.00 = $1.05, totaling $1.10).",
      extraTokenBadge: null,
    },
    {
      id: "vision",
      label: "Vision (<|image|>)",
      userPrompt: "<|image|>What is the cat doing in this photo?",
      imageName: "cat_windowsill.jpg",
      thought: "Inspecting 256 projected image soft tokens: an orange tabby cat with closed eyes is curled up asleep in a warm sunbeam on a wooden windowsill.",
      answer: "An <strong>orange tabby cat</strong> is curled up fast asleep in a warm patch of sunlight on a wooden windowsill.",
      extraTokenBadge: { text: "<|image|>", id: 258880, cls: "token-multi" },
    },
    {
      id: "tool",
      label: "Tool Call (<|tool_call>)",
      userPrompt: "What's the current weather in Tokyo?",
      thought: 'User asks for live weather in Tokyo. I should invoke the declared get_current_weather tool with location="Tokyo".',
      toolDeclSig: "get_current_weather(location: string)",
      toolDeclRaw: `declaration:get_current_weather{description:<span class="token-tool">&lt;|"|&gt;</span>Gets current weather for a city<span class="token-tool">&lt;|"|&gt;</span>,parameters:{properties:{location:{type:<span class="token-tool">&lt;|"|&gt;</span>STRING<span class="token-tool">&lt;|"|&gt;</span>}},required:[<span class="token-tool">&lt;|"|&gt;</span>location<span class="token-tool">&lt;|"|&gt;</span>],type:<span class="token-tool">&lt;|"|&gt;</span>OBJECT<span class="token-tool">&lt;|"|&gt;</span>}}`,
      toolCallSig: 'get_current_weather(location: "Tokyo")',
      toolCallRaw: `call:get_current_weather{location:<span class="token-tool">&lt;|"|&gt;</span>Tokyo<span class="token-tool">&lt;|"|&gt;</span>}`,
      toolRespJson: '{ temperature: 15, weather: "sunny" }',
      toolRespRaw: `response:get_current_weather{temperature:15,weather:<span class="token-tool">&lt;|"|&gt;</span>sunny<span class="token-tool">&lt;|"|&gt;</span>}`,
      answer: "The current weather in Tokyo is <strong>15°C (59°F) and sunny</strong>.",
      extraTokenBadge: { text: "<|tool_call>", id: 48, cls: "token-tool" },
    },
  ];

  const mountPlayground = (el) => {
    let scenIdx = 0;
    let thinkingOn = true;
    let rawView = false;
    let isPlaying = false;
    let playTimer = null;
    let thoughtWordLimit = Infinity;
    let toolStep = 2;
    let answerWordLimit = Infinity;
    let streamedTokCount = 0;

    el.innerHTML = `
      <div class="llm-widget__header">
        <p class="llm-widget__title">Interactive: End-to-End Gemma 4 Chat, Thinking &amp; Tool Stream</p>
      </div>
      <div class="llm-toolbar llm-toolbar--compact" role="group" aria-label="Select prompt scenario">
        <span class="llm-toolbar__label">Scenario</span>
        <div class="llm-toolbar__group" data-role="scenarios"></div>
      </div>
      <div class="llm-toolbar llm-toolbar--split">
        <div class="llm-toolbar__group" role="group" aria-label="View mode and reasoning controls">
          <span class="llm-toolbar__label">View</span>
          <div class="llm-seg" role="group" aria-label="Output format">
            <button type="button" class="llm-seg__btn llm-seg__btn--active" data-view-mode="formatted">Formatted UI</button>
            <button type="button" class="llm-seg__btn" data-view-mode="raw">Raw Token Stream</button>
          </div>
          <span class="llm-toolbar__sep" aria-hidden="true"></span>
          <button type="button" class="llm-switch llm-switch--on" role="switch" aria-checked="true" data-action="toggle-think">
            <span class="llm-switch__track" aria-hidden="true"><span class="llm-switch__thumb"></span></span>
            <span>Thinking <code>&lt;|think|&gt;</code></span>
          </button>
        </div>
        <div class="llm-toolbar__group" role="group" aria-label="Stream playback actions">
          <button type="button" class="llm-action-btn llm-btn--primary" data-action="play-full">▶ Stream Response</button>
        </div>
      </div>
      <div class="llm-panel" data-role="output"></div>
    `;

    const scenEl = el.querySelector('[data-role="scenarios"]');
    const outEl = el.querySelector('[data-role="output"]');
    const thinkBtn = el.querySelector('[data-action="toggle-think"]');
    const playBtn = el.querySelector('[data-action="play-full"]');

    const stopPlayback = () => {
      if (playTimer) {
        clearInterval(playTimer);
        playTimer = null;
      }
      isPlaying = false;
      thoughtWordLimit = Infinity;
      toolStep = 2;
      answerWordLimit = Infinity;
    };

    const sliceWords = (str, limit) => {
      if (limit === Infinity) return str;
      const parts = str.split(" ");
      return parts.slice(0, Math.max(0, limit)).join(" ");
    };

    const render = () => {
      const s = PLAYGROUND_SCENARIOS[scenIdx];
      const isToolScenario = s.id === "tool";
      if (!scenEl.children.length) {
        scenEl.innerHTML = PLAYGROUND_SCENARIOS.map(
          (item, idx) =>
            `<button type="button" class="llm-scenario-btn" data-scen="${idx}">${escapeHtml(item.label)}</button>`
        ).join("");
      }
      scenEl.querySelectorAll("[data-scen]").forEach((btn) => {
        const isAct = Number(btn.dataset.scen) === scenIdx;
        btn.classList.toggle("llm-scenario-btn--active", isAct);
        btn.setAttribute("aria-pressed", String(isAct));
      });

      thinkBtn.classList.toggle("llm-switch--on", thinkingOn);
      thinkBtn.setAttribute("aria-checked", String(thinkingOn));
      el.querySelectorAll("[data-view-mode]").forEach((btn) => {
        const isActive = (btn.dataset.viewMode === "raw") === rawView;
        btn.classList.toggle("llm-seg__btn--active", isActive);
        btn.setAttribute("aria-pressed", String(isActive));
      });
      playBtn.classList.toggle("llm-btn--primary", !isPlaying);
      playBtn.classList.toggle("llm-action-btn--running", isPlaying);
      playBtn.textContent = isPlaying ? "⏸ Stop Streaming" : "▶ Stream Response";

      const thoughtWordsCount = thinkingOn ? s.thought.split(" ").length : 0;
      const isThoughtComplete =
        !thinkingOn || thoughtWordLimit === Infinity || thoughtWordLimit >= thoughtWordsCount;
      const visibleThought = sliceWords(s.thought, thoughtWordLimit);

      const isToolHandshakeDone = !isToolScenario || toolStep >= 2;
      const plainAnswer = s.answer.replace(/<[^>]+>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
      const answerWordsCount = plainAnswer.split(" ").length;
      const isAnswerComplete =
        isThoughtComplete &&
        isToolHandshakeDone &&
        (answerWordLimit === Infinity || answerWordLimit >= answerWordsCount);
      const visibleAnswer =
        !isToolHandshakeDone
          ? ""
          : isAnswerComplete
            ? s.answer
            : escapeHtml(sliceWords(plainAnswer, answerWordLimit));
      const cursorHtml = `<span class="llm-stream-cursor"></span>`;

      const statusBadgeText =
        isPlaying && isToolScenario && isThoughtComplete && toolStep === 1
          ? `⏸ Paused on <|tool_response> — executing ${s.toolCallSig}…`
          : isPlaying
            ? `⚡ Streaming: ${streamedTokCount} tokens · ~84 tok/s`
            : "✓ Generation Complete (stopped on <turn|>)";

      const telemetryBar = `
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.4rem;margin-bottom:0.5rem;padding-bottom:0.4rem;border-bottom:1px solid var(--gemma-border);font-size:0.74rem;color:var(--gemma-text-muted);">
          <span><strong>Chat Template:</strong> <code>&lt;|turn&gt;system</code> · <code>&lt;|turn&gt;user</code> · <code>&lt;|turn&gt;model</code></span>
          <span class="llm-widget__badge">${escapeHtml(statusBadgeText)}</span>
        </div>
      `;

      const systemToolDeclRaw = isToolScenario
        ? `<span class="token-tool">&lt;|tool&gt;</span><code>${s.toolDeclRaw}</code><span class="token-tool">&lt;tool|&gt;</span>`
        : "";
      const userVisionBadgeRaw =
        s.id === "vision" ? `<span class="token-multi" style="margin-right:0.15rem;">&lt;|image|&gt;</span>` : "";
      const userVisionCardFormatted =
        s.id === "vision"
          ? `<div style="display:flex;align-items:center;gap:0.55rem;padding:0.3rem 0.6rem 0.3rem 0.3rem;margin-bottom:0.35rem;width:fit-content;max-width:100%;border-radius:6px;border:1px solid var(--gemma-cat-multi-border);background:var(--gemma-surface-sunken);">
              ${PLAYGROUND_CAT_THUMB_SVG}
              <div style="line-height:1.3;">
                <div style="font-size:0.75rem;font-weight:600;color:var(--md-default-fg-color);">${escapeHtml(s.imageName)}</div>
                <div style="font-size:0.69rem;color:var(--gemma-text-muted);display:flex;align-items:center;gap:0.25rem;margin-top:0.1rem;">
                  <span class="token-multi" style="font-size:0.66rem;">&lt;|image|&gt;</span>
                  <span>256 soft tokens</span>
                </div>
              </div>
            </div>`
          : "";
      const cleanUserPrompt = escapeHtml(s.userPrompt.replace(/^<\|image\|>\s*/, ""));

      if (rawView) {
        let modelStreamRaw = "";
        if (isToolScenario) {
          const thoughtStreamHtml = thinkingOn
            ? `<div><span class="token-think">&lt;|channel&gt;thought</span><code>\\n${escapeHtml(
              visibleThought
            )}${isThoughtComplete ? "\\n" : ""}</code>${isThoughtComplete ? `<span class="token-think">&lt;channel|&gt;</span>` : cursorHtml
            }</div>`
            : "";
          const toolCallRawHtml =
            isThoughtComplete && toolStep >= 1
              ? `<div style="margin-top:0.2rem;"><span class="token-tool">&lt;|tool_call&gt;</span><code>${s.toolCallRaw}</code><span class="token-tool">&lt;tool_call|&gt;</span><span class="token-tool">&lt;|tool_response&gt;</span>${toolStep < 2 ? cursorHtml : ""
              } <span style="font-size:0.7rem;color:var(--gemma-text-muted);font-family:var(--md-text-font-family, sans-serif);white-space:nowrap;">← 1. Halts on &lt;|tool_response&gt;</span></div>`
              : isThoughtComplete
                ? cursorHtml
                : "";
          const toolRespRawHtml =
            isThoughtComplete && toolStep >= 2
              ? `<div style="margin-top:0.2rem;"><code>${s.toolRespRaw}</code><span class="token-tool">&lt;tool_response|&gt;</span> <span style="font-size:0.7rem;color:var(--gemma-text-muted);font-family:var(--md-text-font-family, sans-serif);white-space:nowrap;">← 2. App injects result</span></div>`
              : "";
          const finalAnsRawHtml =
            isThoughtComplete && toolStep >= 2
              ? `<div style="margin-top:0.2rem;"><span>${visibleAnswer}</span>${isAnswerComplete ? `<span class="token-turn">&lt;turn|&gt;</span>` : cursorHtml
              }</div>`
              : "";
          modelStreamRaw = `${thoughtStreamHtml}${toolCallRawHtml}${toolRespRawHtml}${finalAnsRawHtml}`;
        } else {
          const thoughtStreamHtml = thinkingOn
            ? `<span class="token-think">&lt;|channel&gt;thought</span><code>\\n${escapeHtml(
              visibleThought
            )}${isThoughtComplete ? "\\n" : ""}</code>${isThoughtComplete ? `<span class="token-think">&lt;channel|&gt;</span>` : cursorHtml
            }`
            : "";
          const answerStreamHtml = isThoughtComplete
            ? `<span>${visibleAnswer}</span>${isAnswerComplete ? `<span class="token-turn">&lt;turn|&gt;</span>` : cursorHtml
            }`
            : "";
          modelStreamRaw = `${thoughtStreamHtml}${answerStreamHtml}`;
        }

        outEl.innerHTML = `
          ${telemetryBar}
          <div style="font-family:var(--md-code-font-family, monospace);font-size:0.82rem;line-height:1.8;overflow-wrap:anywhere;">
            <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--gemma-text-muted);margin-bottom:0.15rem;">
              Pre-Populated Input Prompt (<code style="text-transform:none;">apply_chat_template</code> prefill — ends at <code style="text-transform:none;">&lt;|turn&gt;model\\n</code>)
            </div>
            <div>
              <span class="token-turn">&lt;|turn&gt;</span><span class="token-role">system</span><code>\\n</code>${
          thinkingOn ? `<span class="token-think">&lt;|think|&gt;</span><code>\\n</code>` : ""
          }<span>${escapeHtml(PLAYGROUND_SYSTEM_PROMPT)}</span>${systemToolDeclRaw}<span class="token-turn">&lt;turn|&gt;</span><code>\\n</code>
            </div>
            <div>
              <span class="token-turn">&lt;|turn&gt;</span><span class="token-role">user</span><code>\\n</code>${userVisionBadgeRaw}<span>${cleanUserPrompt}</span><span class="token-turn">&lt;turn|&gt;</span><code>\\n</code>
            </div>
            <div style="margin-top:0.25rem;padding-top:0.25rem;border-top:1px dashed var(--gemma-border);">
              <div style="font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.04em;color:var(--gemma-text-muted);margin-bottom:0.1rem;">
                Model Turn (Prompt Prefix <code style="text-transform:none;">&lt;|turn&gt;model\\n</code> + Generated Output Stream)
              </div>
              <span class="token-turn">&lt;|turn&gt;</span><span class="token-role">model</span><code>\\n</code>${modelStreamRaw}
            </div>
          </div>
        `;
      } else {
        const systemItemsHtml = [
          thinkingOn
            ? `<span style="display:inline-flex;align-items:center;gap:0.15rem;"><span class="token-think" style="font-size:0.72rem;">&lt;|think|&gt;</span></span>`
            : ``,
          `<span>${escapeHtml(PLAYGROUND_SYSTEM_PROMPT)}</span>`,
          isToolScenario
            ? `<span style="display:inline-flex;align-items:center;flex-wrap:wrap;gap:0.25rem;padding:0.1rem 0.45rem;border-radius:4px;border:1px solid var(--gemma-cat-tool-border);background:var(--gemma-surface-sunken);font-size:0.74rem;">
                <strong>🛠 Declared Tool:</strong> <code>${escapeHtml(s.toolDeclSig)}</code>
                <span class="token-tool" style="font-size:0.68rem;">&lt;|tool&gt;…&lt;tool|&gt;</span>
              </span>`
            : "",
        ]
          .filter(Boolean)
	  .join(``);

        const thoughtBoxHtml = thinkingOn
          ? `<div class="${!isThoughtComplete ? "llm-thinking-box--streaming" : ""}" style="padding:0.5rem 0.75rem;margin-bottom:0.55rem;border:1px solid var(--gemma-cat-think-border);border-left:3px solid var(--gemma-status-warning);background:var(--gemma-surface-sunken);border-radius:4px;font-size:0.8rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.35rem;margin-bottom:0.2rem;">
                <strong>🧠 ${isToolScenario ? "1. " : ""}Thought Process:</strong>
                <span style="font-family:var(--md-code-font-family, monospace);font-size:0.7rem;">
                  <span class="token-think">&lt;|channel&gt;thought</span>…${isThoughtComplete ? `<span class="token-think">&lt;channel|&gt;</span>` : ""}
                </span>
              </div>
              <span style="color:var(--gemma-text-secondary)">${escapeHtml(visibleThought)}</span>${isThoughtComplete ? `` : cursorHtml
          }
            </div>`
          : ``;

        let modelBodyFormatted = "";
        if (isToolScenario) {
          const toolCallCardHtml = !isThoughtComplete
            ? `<div style="font-size:0.78rem;color:var(--gemma-text-muted);font-style:italic;">
                Waiting for model to finish <code>&lt;|channel&gt;thought</code> before emitting <code>&lt;|tool_call&gt;</code>…
              </div>`
            : toolStep >= 1
              ? `<div style="padding:0.5rem 0.75rem;margin-bottom:0.35rem;border:1px solid var(--gemma-cat-tool-border);border-left:3px solid var(--gemma-cat-tool-border);background:var(--gemma-surface-sunken);border-radius:4px;font-size:0.8rem;">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.35rem;margin-bottom:0.25rem;">
                    <span><strong>🔧 2. Model Tool Call Request:</strong> <code>${escapeHtml(s.toolCallSig)}</code></span>
                    <span class="llm-widget__badge" style="font-size:0.68rem;">⏸ Halts on &lt;|tool_response&gt; stop token</span>
                  </div>
                  <div style="font-family:var(--md-code-font-family, monospace);font-size:0.76rem;color:var(--gemma-text-secondary);overflow-x:auto;">
                    <span class="token-tool">&lt;|tool_call&gt;</span><code>${s.toolCallRaw}</code><span class="token-tool">&lt;tool_call|&gt;</span><span class="token-tool">&lt;|tool_response&gt;</span>${toolStep < 2 ? cursorHtml : ""}
                  </div>
                </div>`
              : cursorHtml;

          const toolRespCardHtml =
            isThoughtComplete && toolStep >= 2
              ? `<div style="display:flex;align-items:center;gap:0.4rem;margin:0.2rem 0 0.35rem 0.5rem;font-size:0.74rem;color:var(--gemma-text-muted);">
                  <span>↳ <strong>Your App Runtime</strong> intercepts call, executes <code>${escapeHtml(s.toolCallSig)}</code>, and appends result:</span>
                </div>
                <div style="padding:0.5rem 0.75rem;margin-bottom:0.55rem;border:1px solid var(--gemma-border-strong);border-left:3px solid var(--gemma-accent);background:var(--gemma-surface-sunken);border-radius:4px;font-size:0.8rem;">
                  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.35rem;margin-bottom:0.25rem;">
                    <span><strong>⚡ 3. Injected Tool Response:</strong> <code>${escapeHtml(s.toolRespJson)}</code></span>
                    <span class="llm-widget__badge" style="font-size:0.68rem;">App → Model Context</span>
                  </div>
                  <div style="font-family:var(--md-code-font-family, monospace);font-size:0.76rem;color:var(--gemma-text-secondary);overflow-x:auto;">
                    <code>${s.toolRespRaw}</code><span class="token-tool">&lt;tool_response|&gt;</span>
                  </div>
                </div>`
              : isThoughtComplete && toolStep === 1
                ? `<div style="font-size:0.76rem;color:var(--gemma-text-muted);font-style:italic;margin:0.25rem 0;">
                    ⏳ Waiting for external application to execute <code>${escapeHtml(s.toolCallSig)}</code> and return <code>&lt;tool_response|&gt;</code>…
                  </div>`
                : "";

          const finalAnswerRowHtml =
            isThoughtComplete && toolStep >= 2
              ? `<div style="padding:0.45rem 0.75rem;border:1px solid var(--gemma-border);border-radius:4px;background:var(--gemma-surface);line-height:1.6;">
                  <div style="font-size:0.74rem;font-weight:700;color:var(--gemma-text-muted);margin-bottom:0.15rem;">
                    💬 4. Final Assistant Response (synthesized from tool output)
                  </div>
                  <div>
                    ${visibleAnswer}${isAnswerComplete ? ` <span class="token-turn" style="font-size:0.72rem;">&lt;turn|&gt;</span>` : cursorHtml}
                  </div>
                </div>`
              : "";

          modelBodyFormatted = `${thoughtBoxHtml}${toolCallCardHtml}${toolRespCardHtml}${finalAnswerRowHtml}`;
        } else {
          const answerRowHtml = !isThoughtComplete
            ? `<div style="font-size:0.78rem;color:var(--gemma-text-muted);font-style:italic;">
                Waiting for model to emit closing <code>&lt;channel|&gt;</code> before generating final answer…
              </div>`
            : `<div style="line-height:1.6;">
                ${visibleAnswer}${isAnswerComplete ? ` <span class="token-turn" style="font-size:0.72rem;">&lt;turn|&gt;</span>` : cursorHtml
            }
              </div>`;

          modelBodyFormatted = `${thoughtBoxHtml}${answerRowHtml}`;
        }

        outEl.innerHTML = `
          ${telemetryBar}
          <div style="font-size:0.85rem;">
            <div style="margin-bottom:0.55rem;padding-bottom:0.5rem;border-bottom:1px dashed var(--gemma-border);">
              <div style="margin-bottom:0.2rem;">
                <span class="token-turn" style="font-size:0.72rem;">&lt;|turn&gt;</span> <span class="token-role">system</span>
              </div>
              <div style="padding-left:0.65rem;border-left:2px solid var(--gemma-border);display:flex;align-items:center;flex-wrap:wrap;gap:0.35rem;line-height:1.5;">
                ${systemItemsHtml} <span class="token-turn" style="font-size:0.72rem;">&lt;turn|&gt;</span>
              </div>
            </div>
            <div style="margin-bottom:0.65rem;padding-bottom:0.55rem;border-bottom:1px dashed var(--gemma-border);">
              <div style="margin-bottom:0.25rem;">
                <span class="token-turn" style="font-size:0.72rem;">&lt;|turn&gt;</span> <span class="token-role">user</span>
              </div>
              <div style="padding-left:0.65rem;border-left:2px solid var(--gemma-border);line-height:1.55;">
                ${userVisionCardFormatted}<div>${cleanUserPrompt} <span class="token-turn" style="font-size:0.72rem;">&lt;turn|&gt;</span></div>
              </div>
            </div>
            <div>
              <div style="margin-bottom:0.35rem;">
                <span class="token-turn" style="font-size:0.72rem;">&lt;|turn&gt;</span> <span class="token-role">model</span>
              </div>
              <div style="padding-left:0.65rem;border-left:2px solid var(--gemma-accent);">
                ${modelBodyFormatted}
              </div>
            </div>
          </div>
        `;
      }
    };

    const runStreamTimer = () => {
      const s = PLAYGROUND_SCENARIOS[scenIdx];
      const isToolScenario = s.id === "tool";
      const thoughtWords = thinkingOn ? s.thought.split(" ").length : 0;
      const plainAnswer = s.answer.replace(/<[^>]+>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
      const answerWords = Math.max(2, plainAnswer.split(" ").length);

      playTimer = setInterval(() => {
        if (!el.isConnected) {
          stopPlayback();
          return;
        }
        if (thinkingOn && thoughtWordLimit < thoughtWords) {
          thoughtWordLimit += 2;
          streamedTokCount += 3;
          render();
          return;
        }
        if (isToolScenario && toolStep < 2) {
          toolStep += 1;
          streamedTokCount += toolStep === 1 ? 10 : 12;
          render();
          return;
        }
        if (answerWordLimit < answerWords) {
          answerWordLimit += 2;
          streamedTokCount += 3;
          render();
          return;
        }
        stopPlayback();
        render();
      }, 150);
    };

    const startFullPlay = () => {
      if (isPlaying) {
        stopPlayback();
        render();
        return;
      }
      stopPlayback();
      isPlaying = true;
      thoughtWordLimit = 0;
      toolStep = 0;
      answerWordLimit = 0;
      streamedTokCount = 0;
      runStreamTimer();
      render();
    };

    el.addEventListener("click", (e) => {
      const sBtn = e.target.closest("[data-scen]");
      if (sBtn) {
        stopPlayback();
        scenIdx = Number(sBtn.dataset.scen);
        render();
        return;
      }
      const vBtn = e.target.closest("[data-view-mode]");
      if (vBtn) {
        rawView = vBtn.dataset.viewMode === "raw";
        render();
        return;
      }
      const act = e.target.closest("[data-action]")?.dataset.action;
      if (act === "toggle-think") {
        stopPlayback();
        thinkingOn = !thinkingOn;
        render();
      } else if (act === "toggle-view") {
        rawView = !rawView;
        render();
      } else if (act === "play-full") {
        startFullPlay();
      }
    });

    render();
  };

  /* =====================================================================
     0. OVERVIEW: THE 5-STAGE LLM PIPELINE INTERACTIVE PREVIEW
     Gives readers a concrete visual + live prompt trace before "Start the tour".
     ===================================================================== */
  const OVERVIEW_PIPELINE_STAGES = [
    {
      no: "01",
      stepId: "tokenization",
      shortTitle: "1. Tokenization",
      line1: "Text → Token IDs",
      line2: "(262,144 vocab)",
      traceLabel: "01 · Tokenization",
      traceHtml: `<code>"The cat sat on the"</code> → <span class="llm-ov-tok">The <small>#818</small></span> <span class="llm-ov-tok">▁cat <small>#5866</small></span> <span class="llm-ov-tok">▁sat <small>#2838</small></span> <span class="llm-ov-tok">▁on <small>#580</small></span> <span class="llm-ov-tok">▁the <small>#506</small></span>`,
      cta: "Explore 01 →",
    },
    {
      no: "02",
      stepId: "embeddings",
      shortTitle: "2. Embeddings",
      line1: "IDs → Meaning",
      line2: "Vectors + PLE",
      traceLabel: "02 · Embeddings",
      traceHtml: `Row lookup in <code>262,144 × 1,536</code> table: <code>▁cat (#5866)</code> → <code>[−0.0049, −0.0491, −0.0098, +0.0133, …]</code>`,
      cta: "Explore 02 →",
    },
    {
      no: "03",
      stepId: "attention",
      shortTitle: "3. Layers",
      line1: "Local/Global Attn",
      line2: "+ GeGLU FFN / MoE",
      traceLabel: "03 · Transformer Layers",
      traceHtml: `Hybrid Local &amp; Global Attention (<code>p-RoPE</code> + <code>K=V</code> or KV sharing) links <code>▁the</code> ← <code>▁sat</code> + <code>▁cat</code>, then <strong>GeGLU FFN / MoE</strong> refines each vector`,
      cta: "Explore 03 →",
    },
    {
      no: "04",
      stepId: "sampling",
      shortTitle: "4. Sampling",
      line1: "Logits → Softmax",
      line2: "Temp / K / Top-P",
      traceLabel: "04 · Next-Token Sampling",
      traceHtml: `<code>262,144</code> logits → Softmax: <span class="llm-ov-tok llm-ov-tok--win">▁mat 68% ✓</span> <span class="llm-ov-tok">▁rug 18%</span> <span class="llm-ov-tok">▁floor 9%</span> → samples <strong><code>▁mat</code></strong>`,
      cta: "Explore 04 →",
    },
    {
      no: "05",
      stepId: "putting-it-together",
      shortTitle: "5. Auto-Loop",
      line1: "Append token ID",
      line2: "& repeat from #2",
      traceLabel: "05 · Autoregressive Loop",
      traceHtml: `Appends <strong><code>▁mat (#1756)</code></strong> ↺ loops back to <strong>Stage 2 (Embeddings)</strong> to vectorize the new ID until <span class="token-turn" style="font-size:0.78rem">&lt;turn|&gt;</span>`,
      cta: "Explore 05 →",
    },
  ];

  const mountOverview = (el) => {
    let activeIdx = 0;
    let autoPlay = true;
    let cycleTimer = null;
    let orbRaf = 0;
    const shell = el.closest("[data-llm-shell]");

    el.innerHTML = `
      <div class="llm-overview-pipe" role="region" aria-label="The 5-stage LLM pipeline preview">
        <div class="llm-overview-pipe__head">
          <div class="llm-toolbar__group">
            <span class="llm-overview-pipe__kicker">Interactive: The 5-stage LLM pipeline</span>
            <span class="llm-overview-pipe__hint">· Click any stage to preview, double-click to open</span>
          </div>
          <button type="button" class="llm-action-btn" data-overview-auto style="border-radius:var(--gemma-radius-pill);"></button>
        </div>
        <svg class="llm-overview-pipe__svg" viewBox="0 0 656 122" data-role="ov-svg" aria-label="Flowchart of the 5-stage Gemma 4 LLM pipeline"></svg>
        <div class="llm-overview-pipe__trace" data-role="ov-trace"></div>
      </div>
    `;

    const svgEl = el.querySelector('[data-role="ov-svg"]');
    const traceEl = el.querySelector('[data-role="ov-trace"]');
    const autoBtn = el.querySelector("[data-overview-auto]");

    const boxW = 112;
    const boxH = 68;
    const stepX = 133; // 21px arrow channel between boxes
    const xStart = 6;
    const yTop = 6;
    const yMid = yTop + boxH / 2;

    const isOverviewVisible = () => !shell || !shell.dataset.step || shell.dataset.step === "overview";

    const pointOnPolyline = (pts, t) => {
      let total = 0;
      const segs = [];
      for (let i = 0; i < pts.length - 1; i++) {
        const dx = pts[i + 1][0] - pts[i][0];
        const dy = pts[i + 1][1] - pts[i][1];
        const len = Math.hypot(dx, dy);
        segs.push({ x0: pts[i][0], y0: pts[i][1], dx, dy, len });
        total += len;
      }
      let dist = Math.max(0, Math.min(1, t)) * total;
      for (const s of segs) {
        if (dist <= s.len || s === segs[segs.length - 1]) {
          const u = s.len > 0 ? dist / s.len : 0;
          return [s.x0 + s.dx * u, s.y0 + s.dy * u];
        }
        dist -= s.len;
      }
      return pts[0];
    };

    const getActiveOrbPath = () => {
      if (activeIdx < 4) {
        const x1 = xStart + activeIdx * stepX + boxW + 2;
        const x2 = xStart + (activeIdx + 1) * stepX - 6;
        return {
          pts: [[x1, yMid], [x2, yMid]],
          periodMs: 780,
        };
      }
      const cx5 = xStart + 4 * stepX + boxW / 2;
      const cx2 = xStart + 1 * stepX + boxW / 2;
      return {
        pts: [[cx5, 74], [cx5, 105], [cx2, 105], [cx2, 78]],
        periodMs: 1450,
      };
    };

    const stopOrb = () => {
      if (orbRaf) {
        cancelAnimationFrame(orbRaf);
        orbRaf = 0;
      }
    };

    const startOrb = () => {
      stopOrb();
      if (!isOverviewVisible()) return;
      const tick = (now) => {
        if (!el.isConnected || !isOverviewVisible()) {
          orbRaf = 0;
          return;
        }
        const orb = svgEl.querySelector('[data-role="ov-orb"]');
        const halo = svgEl.querySelector('[data-role="ov-orb-halo"]');
        if (orb && halo) {
          const { pts, periodMs } = getActiveOrbPath();
          const t = (now % periodMs) / periodMs;
          const [px, py] = pointOnPolyline(pts, t);
          orb.setAttribute("cx", px.toFixed(1));
          orb.setAttribute("cy", py.toFixed(1));
          halo.setAttribute("cx", px.toFixed(1));
          halo.setAttribute("cy", py.toFixed(1));
        }
        orbRaf = requestAnimationFrame(tick);
      };
      orbRaf = requestAnimationFrame(tick);
    };

    const stopCycle = () => {
      if (cycleTimer) {
        clearInterval(cycleTimer);
        cycleTimer = null;
      }
    };

    const syncCycle = () => {
      stopCycle();
      startOrb();
      if (!autoPlay || !isOverviewVisible()) {
        if (!isOverviewVisible()) stopOrb();
        return;
      }
      cycleTimer = setInterval(() => {
        if (!el.isConnected) {
          stopCycle();
          stopOrb();
          return;
        }
        if (!isOverviewVisible()) return;
        activeIdx = (activeIdx + 1) % OVERVIEW_PIPELINE_STAGES.length;
        render();
      }, 1800);
    };

    const render = () => {
      autoBtn.textContent = autoPlay ? "⏸ Pause Auto-Play" : "▶ Auto-Play";
      autoBtn.classList.toggle("llm-action-btn--running", autoPlay);
      autoBtn.setAttribute("aria-pressed", String(autoPlay));

      const connectorsSvg = OVERVIEW_PIPELINE_STAGES.slice(0, -1)
        .map((_, idx) => {
          const x1 = xStart + idx * stepX + boxW + 2;
          const x2 = xStart + (idx + 1) * stepX - 7;
          const isPastHop = idx < activeIdx;
          const isActiveHop = idx === activeIdx;
          const lit = isPastHop || isActiveHop;
          const color = lit ? "var(--gemma-accent)" : "var(--gemma-border-strong)";
          return `
            <g>
              <line x1="${x1}" y1="${yMid}" x2="${x2}" y2="${yMid}"
                stroke="${isActiveHop ? "var(--gemma-cat-turn-border)" : color}"
                stroke-width="${lit ? "2.2" : "1.6"}" stroke-linecap="round" />
              ${
                isActiveHop
                  ? `<line x1="${x1}" y1="${yMid}" x2="${x2}" y2="${yMid}"
                      stroke="var(--gemma-accent)" stroke-width="2.4" stroke-linecap="round" class="llm-ov-flow" />`
                  : ""
              }
              <polygon points="${x2 + 6},${yMid} ${x2 - 1.5},${yMid - 4.2} ${x2 - 1.5},${yMid + 4.2}" fill="${color}" />
            </g>
          `;
        })
        .join("");

      const cx5 = xStart + 4 * stepX + boxW / 2;
      const cx2 = xStart + 1 * stepX + boxW / 2;
      const midLoopX = (cx5 + cx2) / 2;
      const loopActive = activeIdx === 4;
      const loopColor = loopActive ? "var(--gemma-accent)" : "var(--gemma-text-muted)";
      const loopPathD = `M ${cx5} 74 L ${cx5} 99 Q ${cx5} 105 ${cx5 - 6} 105 L ${cx2 + 6} 105 Q ${cx2} 105 ${cx2} 99 L ${cx2} 80`;
      const feedbackLoopSvg = `
        <g opacity="${loopActive ? "1" : "0.78"}">
          <path d="${loopPathD}" fill="none" stroke="${loopColor}"
            stroke-width="${loopActive ? "2.2" : "1.4"}"
            class="${loopActive ? "llm-ov-flow--loop" : ""}"
            ${loopActive ? "" : 'stroke-dasharray="4 3"'} />
          <polygon points="${cx2},75 ${cx2 - 4},81.5 ${cx2 + 4},81.5" fill="${loopColor}" />
          <rect x="${midLoopX - 94}" y="95.5" width="188" height="19" rx="9.5"
            fill="${loopActive ? "var(--gemma-cat-turn-bg)" : "var(--gemma-surface)"}"
            stroke="${loopActive ? "var(--gemma-accent)" : "var(--gemma-border-strong)"}" stroke-width="1" />
          <text x="${midLoopX}" y="108" text-anchor="middle" font-size="9.2" font-weight="700"
            font-family="var(--md-code-font-family, monospace)"
            fill="${loopActive ? "var(--gemma-cat-turn-fg)" : "var(--gemma-text-muted)"}">↺ Next step (embed new ID)</text>
        </g>
      `;

      const nodesSvg = OVERVIEW_PIPELINE_STAGES.map((st, idx) => {
        const x = xStart + idx * stepX;
        const cx = x + boxW / 2;
        const isAct = idx === activeIdx;
        const isPast = idx < activeIdx;
        const fill = isAct
          ? "var(--gemma-cat-turn-bg)"
          : isPast
            ? "var(--gemma-surface)"
            : "var(--gemma-surface-sunken)";
        const stroke = isAct
          ? "var(--gemma-accent)"
          : isPast
            ? "var(--gemma-cat-turn-border)"
            : "var(--gemma-border-strong)";
        const titleColor = isAct ? "var(--gemma-cat-turn-fg)" : "var(--md-default-fg-color)";
        return `
          <g class="llm-overview-pipe__node" data-ov-stage="${idx}" role="button" tabindex="0" aria-label="${escapeHtml(st.shortTitle)}: ${escapeHtml(st.line1)} ${escapeHtml(st.line2)}">
            <rect x="${x}" y="${yTop}" width="${boxW}" height="${boxH}" rx="8"
              fill="${fill}" stroke="${stroke}" stroke-width="${isAct ? "2.2" : "1.2"}" />
            <text x="${cx}" y="${yTop + 21}" text-anchor="middle" font-size="11" font-weight="700" fill="${titleColor}">${escapeHtml(st.shortTitle)}</text>
            <text x="${cx}" y="${yTop + 39}" text-anchor="middle" font-size="9.6" font-weight="600" fill="var(--gemma-text-secondary)">${escapeHtml(st.line1)}</text>
            <text x="${cx}" y="${yTop + 54}" text-anchor="middle" font-size="8.8" font-family="var(--md-code-font-family, monospace)" fill="var(--gemma-text-muted)">${escapeHtml(st.line2)}</text>
          </g>
        `;
      }).join("");

      const [initX, initY] = getActiveOrbPath().pts[0];
      const orbSvg = `
        <g pointer-events="none">
          <circle data-role="ov-orb-halo" cx="${initX}" cy="${initY}" r="6.8" fill="var(--gemma-accent)" opacity="0.22" />
          <circle data-role="ov-orb" cx="${initX}" cy="${initY}" r="3.4" fill="var(--gemma-accent-fill)" stroke="var(--gemma-white)" stroke-width="1" />
        </g>
      `;

      svgEl.innerHTML = `${connectorsSvg}${feedbackLoopSvg}${nodesSvg}${orbSvg}`;

      const cur = OVERVIEW_PIPELINE_STAGES[activeIdx];
      traceEl.innerHTML = `
        <div class="llm-overview-pipe__trace-body">
          <strong style="color:var(--md-default-fg-color);margin-right:0.35rem;">${escapeHtml(cur.traceLabel)}:</strong>${cur.traceHtml}
        </div>
        <button type="button" class="llm-overview-pipe__jump" data-llm-step="${cur.stepId}">${escapeHtml(cur.cta)}</button>
      `;
    };

    let lastClickedStage = -1;
    let lastClickTime = 0;

    const openStage = (idx) => {
      if (!shell) return;
      const st = OVERVIEW_PIPELINE_STAGES[idx];
      const tabBtn = st && shell.querySelector(`.llm-shell__tab[data-llm-step="${st.stepId}"]`);
      if (tabBtn) tabBtn.click();
    };

    el.addEventListener("click", (e) => {
      if (e.target.closest("[data-overview-auto]")) {
        autoPlay = !autoPlay;
        if (autoPlay) {
          activeIdx = (activeIdx + 1) % OVERVIEW_PIPELINE_STAGES.length;
        }
        render();
        syncCycle();
        return;
      }
      const node = e.target.closest("[data-ov-stage]");
      if (node) {
        const idx = Number(node.dataset.ovStage);
        const now = Date.now();
        if (idx === lastClickedStage && now - lastClickTime < 450) {
          lastClickedStage = -1;
          lastClickTime = 0;
          openStage(idx);
          return;
        }
        lastClickedStage = idx;
        lastClickTime = now;
        if (activeIdx !== idx || autoPlay) {
          activeIdx = idx;
          autoPlay = false;
          stopCycle();
          render();
          startOrb();
        }
      }
    });

    el.addEventListener("dblclick", (e) => {
      const node = e.target.closest("[data-ov-stage]");
      if (!node) return;
      e.preventDefault();
      openStage(Number(node.dataset.ovStage));
    });

    el.addEventListener("keydown", (e) => {
      const node = e.target.closest("[data-ov-stage]");
      if (!node) return;
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        const idx = Number(node.dataset.ovStage);
        if (activeIdx === idx && !autoPlay) {
          openStage(idx);
        } else {
          activeIdx = idx;
          autoPlay = false;
          stopCycle();
          render();
          startOrb();
        }
      }
    });

    if (shell) {
      shell.addEventListener("llm:step", () => syncCycle());
    }

    render();
    syncCycle();
    if (typeof window !== "undefined") {
      window.requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    }
  };

  /* =====================================================================
     MOUNT DISPATCHER & ZENSICAL INSTANT-NAVIGATION HOOK
     ===================================================================== */
  const WIDGET_MOUNTS = {
    overview: mountOverview,
    tokenizer: mountTokenizer,
    embeddings: mountEmbeddings,
    attention: mountAttention,
    sampling: mountSampling,
    playground: mountPlayground,
  };

  const initLlmTutorial = () => {
    const heroText = document.querySelector('#llm-shell:not(.tuning-lab) [data-llm-hero-text]');
    if (heroText && !heroText.querySelector('[data-llm-widget="overview"]')) {
      const slot = document.createElement("div");
      slot.className = "llm-overview-slot";
      slot.dataset.llmWidget = "overview";
      const actions = heroText.querySelector(".llm-shell__actions");
      if (actions) heroText.insertBefore(slot, actions);
      else heroText.appendChild(slot);
    }

    const containers = document.querySelectorAll("[data-llm-widget]");
    for (const el of containers) {
      if (el.dataset.llmReady === "true") continue;
      const kind = el.dataset.llmWidget;
      if (kind !== "overview") el.classList.add("llm-widget");
      const mountFn = WIDGET_MOUNTS[kind];
      if (typeof mountFn === "function") {
        el.dataset.llmReady = "true";
        mountFn(el);
      }
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLlmTutorial);
  } else {
    initLlmTutorial();
  }
  if (window.document$?.subscribe) {
    window.document$.subscribe(initLlmTutorial);
  }
})();
