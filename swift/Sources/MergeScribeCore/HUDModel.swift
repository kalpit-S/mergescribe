import Foundation

/// The HUD's pure parts: level mapping now; the animation model follows.
public enum HUDModel {
    /// dB above the mic's noise floor that counts as full scale. Speech typically sits
    /// 12-30 dB above the floor, so 30 keeps normal talking in the upper half of the meter.
    static let levelSpanDb = 30.0

    /// A dB reading as 0...1 for the meter: relative to the mic's own floor when it is known,
    /// so the meter reads the same on a quiet built-in mic and a hot condenser; a fixed
    /// -60...0 dB range before then.
    public static func normalizeLevel(db: Double, noiseFloor: Double?) -> Double {
        guard db.isFinite else { return 0 }
        let level: Double
        if let noiseFloor, noiseFloor.isFinite {
            level = (db - noiseFloor) / levelSpanDb
        } else {
            level = (db + 60) / 60
        }
        return max(0, min(1, level))
    }
}
