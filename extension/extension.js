/**
 * Voxscribe GNOME Shell Extension
 *
 * Shows status indicator in the top panel for the Voxscribe speech-to-text daemon.
 * Subscribes to DBus signals for real-time state updates.
 *
 * States:
 * - idle: Indicator hidden
 * - recording: Red mic icon + transcription preview
 * - transcribing: Yellow spinner + "Processing..."
 * - done: Green check + "Copied!" (auto-hides after 5 seconds)
 * - partial: Orange warning + "Partial!" (auto-hides)
 * - error: Red error + "Error!" (auto-hides)
 *
 * Click opens popup with full text and copy button.
 */

import Clutter from "gi://Clutter";
import GLib from "gi://GLib";
import Gio from "gi://Gio";
import GObject from "gi://GObject";
import St from "gi://St";

import { Extension } from "resource:///org/gnome/shell/extensions/extension.js";
import * as Main from "resource:///org/gnome/shell/ui/main.js";
import * as PanelMenu from "resource:///org/gnome/shell/ui/panelMenu.js";
import * as PopupMenu from "resource:///org/gnome/shell/ui/popupMenu.js";

// DBus configuration - must match daemon
const DBUS_NAME = "com.github.frederikb.Voxscribe";
const DBUS_PATH = "/com/github/frederikb/Voxscribe";
const DBUS_INTERFACE = "com.github.frederikb.Voxscribe";

// Icon names for each state
const ICONS = {
  idle: "audio-input-microphone-symbolic",
  recording: "media-record-symbolic",
  transcribing: "emblem-synchronizing-symbolic",
  done: "emblem-ok-symbolic",
  partial: "dialog-warning-symbolic",
  error: "dialog-error-symbolic",
};

// States that auto-hide after delay
const AUTO_HIDE_STATES = {
  done: { label: "Copied!", seconds: 5 },
  partial: { label: "Partial!", seconds: 5 },
  error: { label: "Error!", seconds: 5 },
};

// All possible state classes for cleanup
const STATE_CLASSES = ["recording", "transcribing", "done", "partial", "error"];

const VoxscribeIndicator = GObject.registerClass(
  class VoxscribeIndicator extends PanelMenu.Button {
    _init(settings) {
      super._init(0.0, "Voxscribe Indicator", false);

      this._settings = settings;
      this._fullText = "";
      this._state = "idle";
      this._hideTimeoutId = null;
      this._dbusSignalId = null;

      // Container box for icon + label
      this._box = new St.BoxLayout({
        style_class: "panel-status-menu-box",
      });
      this.add_child(this._box);

      // Icon
      this._icon = new St.Icon({
        icon_name: ICONS.idle,
        style_class: "system-status-icon",
      });
      this._box.add_child(this._icon);

      // Label for transcription preview (CSS handles truncation)
      this._label = new St.Label({
        text: "",
        y_align: Clutter.ActorAlign.CENTER,
        style_class: "voxscribe-label",
      });
      this._box.add_child(this._label);

      // Apply width setting
      this._applyWidthSetting();

      // Watch for setting changes
      this._settingsChangedId = this._settings.connect(
        "changed::label-max-width",
        () => this._applyWidthSetting()
      );

      // Build popup menu
      this._buildMenu();

      // Start hidden
      this.hide();
    }

    /**
     * Apply width setting to label (only max-width is dynamic).
     */
    _applyWidthSetting() {
      const maxWidth = this._settings.get_int("label-max-width");
      this._label.set_style(`max-width: ${maxWidth}px;`);
    }

    /**
     * Build the popup menu with full text and copy button.
     */
    _buildMenu() {
      // Full text display (scrollable)
      this._textItem = new PopupMenu.PopupBaseMenuItem({
        reactive: false,
        can_focus: false,
      });

      // ScrollView for long text (sizing in CSS)
      this._scrollView = new St.ScrollView({
        style_class: "voxscribe-scroll",
        hscrollbar_policy: St.PolicyType.NEVER,
        vscrollbar_policy: St.PolicyType.AUTOMATIC,
      });

      this._textLabel = new St.Label({
        text: "No transcription yet",
        style_class: "voxscribe-popup-text",
      });
      this._textLabel.clutter_text.set_line_wrap(true);
      this._textLabel.clutter_text.set_line_wrap_mode(0); // WORD
      this._textLabel.clutter_text.set_selectable(true);

      // Wrap label in BoxLayout for ScrollView compatibility
      const textBox = new St.BoxLayout({ vertical: true });
      textBox.add_child(this._textLabel);
      this._scrollView.set_child(textBox);
      this._textItem.add_child(this._scrollView);
      this.menu.addMenuItem(this._textItem);

      // Separator
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

      // Copy button
      this._copyItem = new PopupMenu.PopupMenuItem("Copy to Clipboard");
      this._copyItem.connect("activate", () => this._copyToClipboard());
      this.menu.addMenuItem(this._copyItem);
    }

    /**
     * Copy full text to clipboard using native St.Clipboard.
     */
    _copyToClipboard() {
      if (!this._fullText) {
        return;
      }

      try {
        const clipboard = St.Clipboard.get_default();
        clipboard.set_text(St.ClipboardType.CLIPBOARD, this._fullText);
      } catch (e) {
        log(`[Voxscribe] Clipboard copy failed: ${e}`);
      }
    }

    /**
     * Subscribe to DBus signals from daemon.
     */
    connectDbus() {
      try {
        this._dbusSignalId = Gio.DBus.session.signal_subscribe(
          DBUS_NAME,
          DBUS_INTERFACE,
          "StateChanged",
          DBUS_PATH,
          null,
          Gio.DBusSignalFlags.NONE,
          this._onStateChanged.bind(this)
        );
        log("[Voxscribe] DBus signal subscription active");
      } catch (e) {
        log(`[Voxscribe] DBus connection failed: ${e}`);
      }
    }

    /**
     * Fetch initial state from daemon (in case recording is already active).
     */
    fetchInitialStatus() {
      Gio.DBus.session.call(
        DBUS_NAME,
        DBUS_PATH,
        DBUS_INTERFACE,
        "GetStatus",
        null,
        new GLib.VariantType("(ss)"),
        Gio.DBusCallFlags.NONE,
        1000,
        null,
        (connection, result) => {
          try {
            const reply = connection.call_finish(result);
            const [state, text] = reply.deepUnpack();
            log(`[Voxscribe] Initial status: ${state}`);
            this._updateState(state, text);
          } catch (e) {
            // Daemon not running - that's fine, stay hidden
            log(`[Voxscribe] Could not fetch initial status: ${e.message}`);
          }
        }
      );
    }

    /**
     * Unsubscribe from DBus signals.
     */
    disconnectDbus() {
      if (this._dbusSignalId !== null) {
        Gio.DBus.session.signal_unsubscribe(this._dbusSignalId);
        this._dbusSignalId = null;
        log("[Voxscribe] DBus signal subscription removed");
      }
    }

    /**
     * Handle state change signal from daemon.
     */
    _onStateChanged(_connection, _sender, _path, _iface, _signal, params) {
      const [state, text] = params.recursiveUnpack();
      this._updateState(state, text);
    }

    /**
     * Update indicator based on state.
     */
    _updateState(state, text) {
      // Clear any pending hide timeout
      this._clearHideTimeout();

      this._state = state;

      // Update CSS classes for state-based styling
      STATE_CLASSES.forEach((cls) => this._box.remove_style_class_name(cls));
      if (state !== "idle") {
        this._box.add_style_class_name(state);
      }

      if (state === "idle") {
        this._icon.set_icon_name(ICONS.idle);
        this.hide();
        return;
      }

      // Show indicator
      this.show();

      // Update icon
      if (ICONS[state]) {
        this._icon.set_icon_name(ICONS[state]);
      }

      // Handle state-specific behavior
      if (state === "recording") {
        if (text && text.length > 0) {
          this._fullText = text;
          this._textLabel.set_text(text);
          this._label.set_text(text); // CSS handles truncation
        } else {
          // New recording started - clear stale text
          this._fullText = "";
          this._textLabel.set_text("Recording...");
          this._label.set_text("Recording...");
        }
      } else if (state === "transcribing") {
        this._label.set_text("Processing...");
        // Keep _fullText and popup text as-is (show last known text)
      } else if (AUTO_HIDE_STATES[state]) {
        // done, partial, error states
        const config = AUTO_HIDE_STATES[state];
        this._label.set_text(config.label);

        // Update popup with final text if provided
        if (text && text.length > 0) {
          this._fullText = text;
          this._textLabel.set_text(text);
        }

        this._scheduleHide(config.seconds, state);
      }
    }

    /**
     * Schedule auto-hide after delay if still in given state.
     */
    _scheduleHide(seconds, requiredState) {
      this._hideTimeoutId = GLib.timeout_add_seconds(
        GLib.PRIORITY_DEFAULT,
        seconds,
        () => {
          if (this._state === requiredState) {
            this.hide();
          }
          this._hideTimeoutId = null;
          return GLib.SOURCE_REMOVE;
        }
      );
    }

    /**
     * Clear pending hide timeout.
     */
    _clearHideTimeout() {
      if (this._hideTimeoutId !== null) {
        GLib.source_remove(this._hideTimeoutId);
        this._hideTimeoutId = null;
      }
    }

    /**
     * Clean up on destroy.
     */
    destroy() {
      this._clearHideTimeout();
      this.disconnectDbus();

      if (this._settingsChangedId) {
        this._settings.disconnect(this._settingsChangedId);
        this._settingsChangedId = null;
      }

      super.destroy();
    }
  }
);

export default class VoxscribeExtension extends Extension {
  enable() {
    this._settings = this.getSettings();
    this._indicator = new VoxscribeIndicator(this._settings);
    Main.panel.addToStatusArea(this.uuid, this._indicator);
    this._indicator.connectDbus();
    this._indicator.fetchInitialStatus();
    log("[Voxscribe] Extension enabled");
  }

  disable() {
    if (this._indicator) {
      this._indicator.destroy();
      this._indicator = null;
    }
    this._settings = null;
    log("[Voxscribe] Extension disabled");
  }
}
