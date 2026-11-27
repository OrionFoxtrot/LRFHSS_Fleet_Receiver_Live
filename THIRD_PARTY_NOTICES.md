# Third-party notices

This compact Python receiver was written as a specialized reimplementation for the
user's RTL-SDR / STM32WL experiment. It was validated against the behavior of the
Florida State University educational LR-FHSS MATLAB receiver:

https://github.com/jumanamirza/LR-FHSS-receiver

No MATLAB source files from that project are redistributed in this package.

LR-FHSS PHY constants, convolutional-code tables, header interleaving data, and
frequency-hopping behavior correspond to Semtech's LR-FHSS reference driver in
SWDM001. The Semtech source is distributed under the Clear BSD License.

Semtech source:
https://github.com/Lora-net/SWDM001

Semtech copyright notice:
Copyright Semtech Corporation 2021. All rights reserved.

Clear BSD License notice from that source:

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the Semtech corporation nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS
LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The Windows GUI capture helper `rtl_tcp_fleet.exe` uses Osmocom rtl-sdr v2.0.2
under GPL-2.0-or-later. The complete upstream source and license are in
`native/rtl_tcp/rtl-sdr-2.0.2.tar.gz`. The local changes in
`native/rtl_tcp/transfer-size.patch` add a 1 MiB USB transfer option and report
the first callback length. To rebuild on Windows with the station's
`librtlsdr.dll` in the receiver directory, use Python 3.12 or newer, Git and
64-bit MinGW-w64 GCC with POSIX threads on PATH, then run
`python native/rtl_tcp/build.py`. Compatible separately installed `rtl_tcp.exe` helpers are also supported.

[Osmocom rtl-sdr v2.0.2](https://github.com/osmocom/rtl-sdr/tree/v2.0.2)

## Windows runtime DLL audit

The origin of the three DLLs below is identified as Osmocom's
`rtl-sdr-64bit-20260906.zip`. The maintainer confirmed that its extracted DLLs
match these SHA-256 values on **2026-11-27**. The
[Osmocom directory listing](https://ftp.osmocom.org/binaries/windows/rtl-sdr/)
dates the archive **2026-09-06** and lists its size as **384.2K**.
The DLLs are **not redistributed** by this project. Users download that archive
themselves and place the three DLLs beside `rtl_tcp_fleet.exe`.

| File | Observed SHA-256 | Redistribution status |
|---|---|---|
| `librtlsdr.dll` | `737e76323feb82377c6289d5307241cbf6dbff1d9792b6861b5fd5560d9dd2ec` | Osmocom rtl-sdr-64bit-20260906 (hash-matched); not redistributed |
| `libusb-1.0.dll` | `1c36076c75ded5ef34c5e93494988517a83f1818f2bc2f0399f5d3519433f73d` | Osmocom rtl-sdr-64bit-20260906 (hash-matched); not redistributed |
| `libwinpthread-1.dll` | `80654d8155b1ba2b58938f4524caa5404a4cf64d1ccd7d74149bbde4e938b974` | Osmocom rtl-sdr-64bit-20260906 (hash-matched); not redistributed |

Upstream projects state the following licence terms:

- `librtlsdr.dll`: GPL-2.0-or-later ([rtl-sdr source notice](https://github.com/osmocom/rtl-sdr/blob/master/src/librtlsdr.c)).
- `libusb-1.0.dll`: LGPL-2.1-or-later ([libusb source notice](https://github.com/libusb/libusb/blob/master/libusb/core.c)).
- `libwinpthread-1.dll`: MIT and BSD-3-Clause terms ([winpthreads COPYING](https://github.com/mingw-w64/mingw-w64/blob/master/mingw-w64-libraries/winpthreads/COPYING)).

The release metadata keeps the exporter's `status: open` omission gate. Origin
identification does not opt these DLLs into redistribution. Bundling them later
requires repository-held licence/origin evidence and matching source availability.
`WINDOWS_DLLS.json` records the omissions; `ZIP_MANIFEST.json` hashes the ZIP
payload. `rtl_tcp_fleet.exe` and its included source retain the GPL notice above.
