#!/bin/bash
# Run the tests with the Command Line Tools toolchain (Swift 6.2, macOS 26 SDK), which the
# app needs for SpeechAnalyzer; Xcode 16 here only has the macOS 15 SDK.
#
# Swift Testing ships with the Command Line Tools but outside SwiftPM's search path, and its
# Foundation overlay (_Testing_Foundation) is missing its module there, so cross-import
# overlays are switched off.
set -euo pipefail
cd "$(dirname "$0")/.."
FW=/Library/Developer/CommandLineTools/Library/Developer/Frameworks
LIB=/Library/Developer/CommandLineTools/Library/Developer/usr/lib
exec env -u DEVELOPER_DIR swift test --scratch-path .build-clt \
    -Xswiftc -F -Xswiftc "$FW" -Xswiftc -Xfrontend -Xswiftc -disable-cross-import-overlays \
    -Xlinker -F -Xlinker "$FW" -Xlinker -rpath -Xlinker "$FW" -Xlinker -rpath -Xlinker "$LIB" "$@"
