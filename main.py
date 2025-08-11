import os
import signal
import subprocess
import sys
import threading
import time

import rumps
from pynput import keyboard

import transcriber
from config_manager import ConfigManager


class TranscriberApp(rumps.App):
    def __init__(self):
        super(TranscriberApp, self).__init__("🎤")
        self.state = transcriber.get_global_state()
        self.state.turbo_mode = "--turbo" in sys.argv
        
        transcriber.configure_pyautogui()
        
        self.turbo_item = rumps.MenuItem("Turbo Mode", callback=self.toggle_turbo_mode)
        self.turbo_item.state = int(self.state.turbo_mode)

        cfg = ConfigManager()
        self.trigger_key = getattr(keyboard.Key, cfg.get_value("TRIGGER_KEY") or "alt_r", None)
        self.trigger_label_item = rumps.MenuItem(
            f"Trigger Key: {cfg.get_value('TRIGGER_KEY') or 'alt_r'}"
        )
        self.menu = [
            self.trigger_label_item,
            self.turbo_item,
            None,
            "Settings...",
            None,
            "Reset State"
        ]
        
        self.setup_keyboard_listener()
        
        self.update_timer = rumps.Timer(self.update_status, 0.2)
        self.update_timer.start()
        
        try:
            if hasattr(self, "quit_button") and self.quit_button is not None:
                self.quit_button.set_callback(self.quit_app_clicked)
        except Exception:
            pass

        cfg = ConfigManager()
        print("MergeScribe UI started! (from main.py)")
        try:
            rumps.notification("MergeScribe", "Started", f"Look for 🎙️ in menu bar. Trigger: {cfg.get_value('TRIGGER_KEY') or 'alt_r'}")
        except Exception as e:
            print(f"Notification failed (this is normal in some environments): {e}")
            print(f"Look for 🎙️ in menu bar. Trigger: {cfg.get_value('TRIGGER_KEY') or 'alt_r'}")

    def setup_keyboard_listener(self):
        """Set up keyboard listener."""
        self.shift_pressed = False
        self.ctrl_pressed = False
        self.trigger_key_pressed = False
        
        def on_press(key):
            try:
                self._handle_key_press(key)
            except Exception as e:
                print(f"Error in on_press: {e}")

        def on_release(key):
            try:
                self._handle_key_release(key)
            except Exception as e:
                print(f"Error in on_release: {e}")
                if ConfigManager().get_value("DEBUG_MODE"):
                    print("[DEBUG] Force resetting state due to on_release error")
                transcriber.reset_state()
        
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        
        listener_thread = threading.Thread(target=self.listener.start, daemon=True)
        listener_thread.start()
    
    def _handle_key_press(self, key):
        """Handle individual key press events"""
        if self._is_modifier_key(key):
            self._update_modifier_state(key, True)
        elif key == self.trigger_key:
            self._handle_trigger_press()
    
    def _handle_key_release(self, key):
        """Handle individual key release events"""
        if self._is_modifier_key(key):
            self._update_modifier_state(key, False)
        elif key == self.trigger_key:
            self._handle_trigger_release()
        elif self._is_reset_combination(key):
            self._handle_reset_combinations(key)
    
    def _is_modifier_key(self, key):
        """Check if key is a modifier key"""
        return key in [
            keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
            keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r
        ]
    
    def _update_modifier_state(self, key, pressed):
        """Update modifier key state"""
        if key in [keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r]:
            self.shift_pressed = pressed
        elif key in [keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r]:
            self.ctrl_pressed = pressed
    
    def _handle_trigger_press(self):
        """Handle trigger key press"""
        self.trigger_key_pressed = True
        
        # Block if already busy to prevent hardware conflicts
        if not self.state.recording_in_progress and not self.state.audio_hardware_busy:
            if ConfigManager().get_value("DEBUG_MODE"):
                print(f"[DEBUG] Starting recording... State: recording={self.state.recording_in_progress}")
            
            self.state.recording_in_progress = True
            transcriber.start_recording()
            
            # Detect text selection in parallel
            threading.Thread(target=transcriber.detect_and_store_selected_text, daemon=True).start()
        else:
            if ConfigManager().get_value("DEBUG_MODE"):
                reason = "already recording" if self.state.recording_in_progress else "audio hardware busy"
                print(f"[DEBUG] Key pressed but {reason} - ignoring")
    
    def _handle_trigger_release(self):
        """Handle trigger key release"""
        self.trigger_key_pressed = False
        if self.state.recording_in_progress:
            if ConfigManager().get_value("DEBUG_MODE"):
                print(f"[DEBUG] Stopping recording... State: recording={self.state.recording_in_progress}")
            transcriber.stop_recording_and_process()
    
    def _is_reset_combination(self, key):
        """Check if key is part of a reset combination"""
        return (
            (key == keyboard.Key.esc and self.shift_pressed) or
            (key == keyboard.Key.f12 and self.shift_pressed) or
            (key == keyboard.KeyCode.from_char('r') and self.ctrl_pressed and self.shift_pressed)
        )
    
    def _handle_reset_combinations(self, key):
        """Handle reset key combinations"""
        if key == keyboard.Key.esc and self.shift_pressed:
            if ConfigManager().get_value("DEBUG_MODE"):
                print("[DEBUG] Manual reset requested")
            transcriber.reset_state()
        elif ((key == keyboard.Key.f12 and self.shift_pressed) or 
              (key == keyboard.KeyCode.from_char('r') and self.ctrl_pressed and self.shift_pressed)):
            if ConfigManager().get_value("DEBUG_MODE"):
                print("[DEBUG] EMERGENCY RESET - Force killing all processing")
            self.force_reset()

    def force_reset(self):
        """Emergency reset function to recover from hangs"""
        try:
            print("[EMERGENCY] Force resetting all state...")
            
            with self.state.processing_lock:
                self._cleanup_threads()
                self._reset_state_flags()
                self._force_close_stream()
                        
            try:
                rumps.notification("MergeScribe", "EMERGENCY RESET", "Forced system reset completed")
            except Exception:
                pass
            print("[EMERGENCY] Reset completed")
        except Exception as e:
            print(f"[EMERGENCY] Error during force reset: {e}")
            self._last_resort_cleanup()
    
    def _cleanup_threads(self):
        """Clean up active threads during emergency reset"""
        if hasattr(self.state, 'active_threads'):
            alive_count = len([t for t in self.state.active_threads if t.is_alive()])
            if alive_count > 0:
                print(f"[EMERGENCY] Found {alive_count} active threads - clearing list")
            self.state.active_threads.clear()
    
    def _reset_state_flags(self):
        """Reset all state flags during emergency reset"""
        self.state.recording_in_progress = False
        self.state.audio_hardware_busy = False
        self.state.audio_buffer.clear()
        self.state.recording_start_time = None
    
    def _force_close_stream(self):
        """Aggressively close audio stream during emergency reset"""
        if self.state.stream:
            try:
                self.state.stream.stop()
                time.sleep(0.1)
                self.state.stream.close()
                time.sleep(0.1)
            except Exception as stream_error:
                print(f"[EMERGENCY] Stream cleanup error: {stream_error}")
            finally:
                self.state.stream = None
    
    def _last_resort_cleanup(self):
        """If emergency reset fails, try basic cleanup"""
        try:
            self.state.recording_in_progress = False
            self.state.audio_hardware_busy = False
            self.state.stream = None
        except Exception:
            pass

    def update_status(self, _):
        try:
            # Refresh trigger key setting dynamically
            cfg = ConfigManager()
            new_trigger = getattr(keyboard.Key, cfg.get_value("TRIGGER_KEY") or "alt_r", None)
            if new_trigger != getattr(self, 'trigger_key', None):
                self.trigger_key = new_trigger
                # Update menu label via stored reference
                if hasattr(self, 'trigger_label_item') and self.trigger_label_item is not None:
                    self.trigger_label_item.title = f"Trigger Key: {cfg.get_value('TRIGGER_KEY') or 'alt_r'}"
            current_title = self.title
            new_title = ""
            
            # Check for stuck recording (key released but recording still active)
            if self.state.recording_in_progress:
                if hasattr(self.state, 'recording_start_time'):
                    recording_duration = time.time() - self.state.recording_start_time
                    
                    # ONLY stop if trigger key is no longer pressed (missed release event)
                    # Allow unlimited recording time while key is held
                    if not self.trigger_key_pressed and recording_duration > 1.0:  # Give 1 second grace period
                        print(f"[WARNING] Recording active but trigger key not pressed - missed release event after {recording_duration:.1f}s")
                        print("[DEBUG] Force stopping recording due to missed key release")
                        transcriber.stop_recording_and_process()
                        return
                new_title = "🔴"
            elif hasattr(self.state, 'active_threads') and any(t.is_alive() for t in self.state.active_threads):
                # Check if processing is taking too long
                active_count = len([t for t in self.state.active_threads if t.is_alive()])
                if active_count > 2:  # Keep this - too many processing threads is still a problem
                    print(f"[WARNING] Too many active processing threads ({active_count}) - auto-resetting")
                    self.force_reset()
                    return
                new_title = "⚡"
            else:
                new_title = "🎤"
            
            if current_title != new_title:
                self.title = new_title
        except Exception as e:
            print(f"Error updating status: {e}")
            if self.title != "❓":
                 self.title = "❓"

    @rumps.clicked("Settings...")
    def settings_clicked(self, _):
        """Open the settings dialog"""
        try:
            # Launch the settings dialog in a separate process to avoid thread conflicts
            current_dir = os.path.dirname(os.path.abspath(__file__))
            settings_script = os.path.join(current_dir, "settings_dialog.py")
            subprocess.Popen([sys.executable, settings_script, "--flet"])
        except Exception as e:
            print(f"Error opening settings: {e}")
            rumps.notification("MergeScribe", "Error", "Could not open settings dialog")

    @rumps.clicked("Reset State")
    def reset_clicked(self, _):
        transcriber.reset_state()
        rumps.notification("MergeScribe", "Reset", "State has been reset.")

    def quit_app_clicked(self, _):
        self.clean_exit()
        rumps.quit_application()

    def toggle_turbo_mode(self, sender):
        """Toggle Turbo Mode from menu."""
        try:
            self.state.turbo_mode = not self.state.turbo_mode
            sender.state = int(self.state.turbo_mode)
        except Exception as e:
            print(f"Error toggling Turbo Mode: {e}")

    def clean_exit(self):
        print("Cleaning up resources...")
        try:
            if hasattr(self, 'update_timer') and getattr(self.update_timer, 'is_alive', lambda: False)():
                self.update_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'listener') and getattr(self.listener, 'is_alive', lambda: False)():
                self.listener.stop()
        except Exception:
            pass
        transcriber.cleanup()
        print("Cleanup complete.")

def main():
    app = TranscriberApp()

    def signal_handler(sig, frame):
        print(f"Signal {sig} received, exiting...")
        app.clean_exit()
        rumps.quit_application()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.run()
    except Exception as e:
        print(f"Rumps app error: {e}")
        app.clean_exit()
    finally:
        print("Application has exited.")

if __name__ == "__main__":
    main()
