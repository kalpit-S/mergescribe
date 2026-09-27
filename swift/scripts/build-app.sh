#!/bin/bash
# Build MergeScribe.app from the Swift package.
#
#   swift/scripts/build-app.sh            # release build into swift/build/MergeScribe.app
#
# Signing: macOS ties Microphone, Input Monitoring and Accessibility permission to the app's
# signature. An ad-hoc signature changes with every build, so each rebuild would ask again.
# Signing with a stable identity keeps them: create a self-signed "Code Signing" certificate
# named "MergeScribe Dev" once (Keychain Access > Certificate Assistant > Create a Certificate),
# or set MERGESCRIBE_SIGN_IDENTITY to one you have. Without one, the build is signed ad hoc.
set -euo pipefail
cd "$(dirname "$0")/.."
# The Command Line Tools toolchain (Swift 6.2, macOS 26 SDK): SpeechAnalyzer needs the
# macOS 26 SDK, and the Xcode here (16.4) only has macOS 15's.
unset DEVELOPER_DIR

swift build -c release --product MergeScribe --scratch-path .build-clt

APP=build/MergeScribe.app
VERSION=$(sed -n 's/.*static let version = "\(.*\)".*/\1/p' Sources/MergeScribeCore/Types.swift)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp .build-clt/release/MergeScribe "$APP/Contents/MacOS/MergeScribe"
cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>io.github.kalpit-s.mergescribe</string>
    <key>CFBundleName</key><string>MergeScribe</string>
    <key>CFBundleDisplayName</key><string>MergeScribe</string>
    <key>CFBundleExecutable</key><string>MergeScribe</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>LSMinimumSystemVersion</key><string>14.0</string>
    <key>LSUIElement</key><true/>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>MergeScribe records while you hold the trigger key, to turn what you say into text.</string>
</dict>
</plist>
PLIST

IDENTITY="${MERGESCRIBE_SIGN_IDENTITY:-MergeScribe Dev}"
if security find-identity -v -p codesigning 2>/dev/null | grep -q "\"$IDENTITY\""; then
    codesign --force --sign "$IDENTITY" "$APP"
    echo "Signed with \"$IDENTITY\""
else
    codesign --force --sign - "$APP"
    echo "Signed ad hoc (no \"$IDENTITY\" identity): permissions will be asked for again after each build"
fi
echo "Built $APP"
