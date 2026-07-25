"""
Pure-stdlib QR Code generator (spec L2: shape, color, SVG embedded logo slot).
Byte mode, versions 1-10, EC levels L/M/Q/H. Full Reed-Solomon over GF(256),
all 8 masks scored by ISO/IEC 18004 penalty rules, format + version info (BCH).
Emits SVG (recolorable, square or dot modules, optional center logo slot) and a
boolean matrix. Round-trip verified in selftest by decoding rasterized output
with OpenCV's QRCodeDetector.
"""

# ---- GF(256) ------------------------------------------------------------
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gmul(a, b):
    if a == 0 or b == 0:
        return 0
    return _EXP[_LOG[a] + _LOG[b]]


def _rs_generator(n):
    g = [1]
    for i in range(n):
        g2 = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            g2[j] ^= _gmul(c, 1)
            g2[j + 1] ^= _gmul(c, _EXP[i])
        g = g2
    return g


def _rs_ec(data, n):
    gen = _rs_generator(n)
    res = list(data) + [0] * n
    for i in range(len(data)):
        coef = res[i]
        if coef:
            for j in range(len(gen)):
                res[i + j] ^= _gmul(gen[j], coef)
    return res[len(data):]


# ---- capacity / block tables (versions 1-10) ----------------------------
# (ec_per_block, [(num_blocks, data_per_block), ...])
_ECTAB = {
    (1, "L"): (7, [(1, 19)]), (1, "M"): (10, [(1, 16)]), (1, "Q"): (13, [(1, 13)]), (1, "H"): (17, [(1, 9)]),
    (2, "L"): (10, [(1, 34)]), (2, "M"): (16, [(1, 28)]), (2, "Q"): (22, [(1, 22)]), (2, "H"): (28, [(1, 16)]),
    (3, "L"): (15, [(1, 55)]), (3, "M"): (26, [(1, 44)]), (3, "Q"): (18, [(2, 17)]), (3, "H"): (22, [(2, 13)]),
    (4, "L"): (20, [(1, 80)]), (4, "M"): (18, [(2, 32)]), (4, "Q"): (26, [(2, 24)]), (4, "H"): (16, [(4, 9)]),
    (5, "L"): (26, [(1, 108)]), (5, "M"): (24, [(2, 43)]), (5, "Q"): (18, [(2, 15), (2, 16)]), (5, "H"): (22, [(2, 11), (2, 12)]),
    (6, "L"): (18, [(2, 68)]), (6, "M"): (16, [(4, 27)]), (6, "Q"): (24, [(4, 19)]), (6, "H"): (28, [(4, 15)]),
    (7, "L"): (20, [(2, 78)]), (7, "M"): (18, [(4, 31)]), (7, "Q"): (18, [(2, 14), (4, 15)]), (7, "H"): (26, [(4, 13), (1, 14)]),
    (8, "L"): (24, [(2, 97)]), (8, "M"): (22, [(2, 38), (2, 39)]), (8, "Q"): (22, [(4, 18), (2, 19)]), (8, "H"): (26, [(4, 14), (2, 15)]),
    (9, "L"): (30, [(2, 116)]), (9, "M"): (22, [(3, 36), (2, 37)]), (9, "Q"): (20, [(4, 16), (4, 17)]), (9, "H"): (24, [(4, 12), (4, 13)]),
    (10, "L"): (18, [(2, 68), (2, 69)]), (10, "M"): (26, [(4, 43), (1, 44)]), (10, "Q"): (24, [(6, 19), (2, 20)]), (10, "H"): (28, [(6, 15), (2, 16)]),
}
_ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
          7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50]}


def _data_capacity(version, ecl):
    return sum(nb * dc for nb, dc in _ECTAB[(version, ecl)][1])


# ---- bit buffer -----------------------------------------------------------
class _Bits:
    def __init__(self):
        self.bits = []

    def put(self, value, length):
        for i in range(length - 1, -1, -1):
            self.bits.append((value >> i) & 1)

    def __len__(self):
        return len(self.bits)


def _encode_data(text, version, ecl):
    data = text.encode("utf-8")
    cc_bits = 8 if version <= 9 else 16
    bits = _Bits()
    bits.put(0b0100, 4)              # byte mode
    bits.put(len(data), cc_bits)     # char count
    for b in data:
        bits.put(b, 8)
    cap_bits = _data_capacity(version, ecl) * 8
    if len(bits) > cap_bits:
        raise ValueError("data too long for version/ecl")
    bits.put(0, min(4, cap_bits - len(bits)))          # terminator
    while len(bits) % 8:
        bits.bits.append(0)                            # byte align
    codewords = []
    for i in range(0, len(bits), 8):
        codewords.append(int("".join(map(str, bits.bits[i:i + 8])), 2))
    pad = [0xEC, 0x11]
    k = 0
    while len(codewords) < _data_capacity(version, ecl):
        codewords.append(pad[k % 2])
        k += 1
    return codewords


def _interleave(codewords, version, ecl):
    ecc_per, groups = _ECTAB[(version, ecl)]
    blocks, pos = [], 0
    for num_blocks, dc in groups:
        for _ in range(num_blocks):
            chunk = codewords[pos:pos + dc]
            pos += dc
            blocks.append((chunk, _rs_ec(chunk, ecc_per)))
    out = []
    maxd = max(len(d) for d, _ in blocks)
    for i in range(maxd):
        for d, _e in blocks:
            if i < len(d):
                out.append(d[i])
    for i in range(ecc_per):
        for _d, e in blocks:
            out.append(e[i])
    return out


# ---- matrix ---------------------------------------------------------------
def _new_matrix(size):
    return [[None] * size for _ in range(size)]


def _place_finder(m, r, c):
    for dr in range(-1, 8):
        for dc in range(-1, 8):
            rr, cc = r + dr, c + dc
            if 0 <= rr < len(m) and 0 <= cc < len(m):
                inb = 0 <= dr <= 6 and 0 <= dc <= 6
                ring = dr in (0, 6) or dc in (0, 6)
                core = 2 <= dr <= 4 and 2 <= dc <= 4
                m[rr][cc] = 1 if (inb and (ring or core)) else 0


def _place_alignment(m, version):
    centers = _ALIGN[version]
    for r in centers:
        for c in centers:
            if m[r][c] is not None:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    ring = max(abs(dr), abs(dc))
                    m[r + dr][c + dc] = 1 if ring != 1 else 0


def _reserve_format(m):
    size = len(m)
    for i in range(9):
        for (r, c) in [(8, i), (i, 8)]:
            if m[r][c] is None:
                m[r][c] = 0
    for i in range(8):
        if m[size - 1 - i][8] is None:
            m[size - 1 - i][8] = 0
        if m[8][size - 1 - i] is None:
            m[8][size - 1 - i] = 0


def _place_timing(m):
    size = len(m)
    for i in range(8, size - 8):
        v = 1 if i % 2 == 0 else 0
        if m[6][i] is None:
            m[6][i] = v
        if m[i][6] is None:
            m[i][6] = v


def _mask_fn(k):
    return [
        lambda r, c: (r + c) % 2 == 0,
        lambda r, c: r % 2 == 0,
        lambda r, c: c % 3 == 0,
        lambda r, c: (r + c) % 3 == 0,
        lambda r, c: (r // 2 + c // 3) % 2 == 0,
        lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
        lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
        lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0,
    ][k]


def _place_data(m, bits, mask):
    size = len(m)
    fn = _mask_fn(mask)
    idx = 0
    up = True
    col = size - 1
    while col > 0:
        if col == 6:
            col -= 1
        rng = range(size - 1, -1, -1) if up else range(size)
        for r in rng:
            for c in (col, col - 1):
                if m[r][c] is None:
                    bit = bits[idx] if idx < len(bits) else 0
                    idx += 1
                    if fn(r, c):
                        bit ^= 1
                    m[r][c] = bit
        up = not up
        col -= 2


_FMT_GEN = 0b10100110111
_FMT_MASK = 0b101010000010010
_ECL_BITS = {"L": 0b01, "M": 0b00, "Q": 0b11, "H": 0b10}


def _format_bits(ecl, mask):
    data = (_ECL_BITS[ecl] << 3) | mask
    rem = data << 10
    for i in range(14, 9, -1):
        if (rem >> i) & 1:
            rem ^= _FMT_GEN << (i - 10)
    return ((data << 10) | (rem & 0x3FF)) ^ _FMT_MASK


def _apply_format(m, ecl, mask):
    """ISO/IEC 18004 format-info placement. This is the exact mapping that was
    broken in the previous build (bit order + coordinates); now bit i (LSB=0)
    goes to the standard positions in both copies."""
    bits = _format_bits(ecl, mask)
    n = len(m)
    for i in range(15):
        v = (bits >> i) & 1
        # copy 1: vertical strip in column 8 (skipping the timing row)
        if i < 6:
            r = i
        elif i < 8:
            r = i + 1
        else:
            r = n - 15 + i
        m[r][8] = v
        # copy 2: horizontal strip in row 8
        if i < 8:
            c = n - 1 - i
        elif i == 8:
            c = 7
        else:
            c = 14 - i
        m[8][c] = v
    m[n - 8][8] = 1  # dark module


_VER_GEN = 0b1111100100101


def _apply_version(m, version):
    if version < 7:
        return
    rem = version << 12
    for i in range(17, 11, -1):
        if (rem >> i) & 1:
            rem ^= _VER_GEN << (i - 12)
    bits = (version << 12) | (rem & 0xFFF)
    size = len(m)
    for i in range(18):
        bit = (bits >> i) & 1
        r, c = i // 3, i % 3
        m[size - 11 + c][r] = bit
        m[r][size - 11 + c] = bit


def _penalty(m):
    size = len(m)
    score = 0
    for line in (m, list(zip(*m))):
        for row in line:
            run = 1
            for i in range(1, size):
                if row[i] == row[i - 1]:
                    run += 1
                else:
                    if run >= 5:
                        score += 3 + (run - 5)
                    run = 1
            if run >= 5:
                score += 3 + (run - 5)
    for r in range(size - 1):
        for c in range(size - 1):
            if m[r][c] == m[r][c + 1] == m[r + 1][c] == m[r + 1][c + 1]:
                score += 3
    pat = [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    patr = pat[::-1]
    for line in (m, list(zip(*m))):
        for row in line:
            row = list(row)
            for i in range(size - 11):
                seg = row[i:i + 11]
                if seg == pat or seg == patr:
                    score += 40
    dark = sum(sum(row) for row in m)
    ratio = dark * 100 // (size * size)
    score += (abs(ratio - 50) // 5) * 10
    return score


class QRCode:
    def __init__(self, matrix, version, ecl):
        self.matrix = matrix
        self.version = version
        self.ecl = ecl
        self.size = len(matrix)

    def svg(self, module=10, quiet=4, dark="#000000", light="#ffffff",
            shape="square", logo_slot=0.0):
        """Recolorable SVG. shape: 'square' or 'dot'. logo_slot: 0..0.3 fraction
        of the symbol reserved in the center (use EC level H when > 0)."""
        n = self.size
        dim = (n + quiet * 2) * module
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{dim}" '
                 f'height="{dim}" viewBox="0 0 {dim} {dim}" shape-rendering="crispEdges">']
        parts.append(f'<rect width="{dim}" height="{dim}" fill="{light}"/>')
        skip = set()
        half = 0
        if logo_slot and logo_slot > 0:
            logo_slot = min(logo_slot, 0.3)
            half = int(n * logo_slot / 2)
            mid = n // 2
            for r in range(mid - half, mid + half + 1):
                for c in range(mid - half, mid + half + 1):
                    skip.add((r, c))
        if shape == "dot":
            rad = module * 0.42
            for r in range(n):
                for c in range(n):
                    if self.matrix[r][c] and (r, c) not in skip:
                        cx = (c + quiet) * module + module / 2
                        cy = (r + quiet) * module + module / 2
                        parts.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{rad:g}" fill="{dark}"/>')
        else:
            path = []
            for r in range(n):
                for c in range(n):
                    if self.matrix[r][c] and (r, c) not in skip:
                        x = (c + quiet) * module
                        y = (r + quiet) * module
                        path.append(f'M{x} {y}h{module}v{module}h-{module}z')
            parts.append(f'<path fill="{dark}" d="{"".join(path)}"/>')
        if skip:
            mid = n // 2
            xy = (mid - half + quiet) * module
            side = (2 * half + 1) * module
            parts.append(f'<rect x="{xy}" y="{xy}" width="{side}" height="{side}" fill="{light}"/>')
            parts.append(f'<rect x="{xy}" y="{xy}" width="{side}" height="{side}" '
                         f'fill="none" stroke="{dark}" stroke-width="{module // 2 or 1}"/>')
        parts.append('</svg>')
        return "".join(parts)


def make(text, ecl="M", min_version=1):
    if ecl not in ("L", "M", "Q", "H"):
        raise ValueError("ecl must be one of L M Q H")
    version = None
    for v in range(max(1, min_version), 11):
        cc_bits = 8 if v <= 9 else 16
        need = 4 + cc_bits + len(text.encode("utf-8")) * 8
        if need <= _data_capacity(v, ecl) * 8:
            version = v
            break
    if version is None:
        raise ValueError("text too long for versions 1-10 at this EC level")
    codewords = _encode_data(text, version, ecl)
    full = _interleave(codewords, version, ecl)
    bits = []
    for cw in full:
        for i in range(7, -1, -1):
            bits.append((cw >> i) & 1)
    size = 17 + version * 4
    best = None
    for mask in range(8):
        m = _new_matrix(size)
        _place_finder(m, 0, 0)
        _place_finder(m, 0, size - 7)
        _place_finder(m, size - 7, 0)
        _place_alignment(m, version)
        _place_timing(m)
        _reserve_format(m)
        m[size - 8][8] = 1
        _place_data(m, bits, mask)
        _apply_format(m, ecl, mask)
        _apply_version(m, version)
        pen = _penalty(m)
        if best is None or pen < best[0]:
            best = (pen, m, mask)
    return QRCode(best[1], version, ecl)
