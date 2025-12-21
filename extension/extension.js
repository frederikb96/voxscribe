/**
 * Voxscribe GNOME Shell Extension
 *
 * Shows status indicator in the top panel for the Voxscribe speech-to-text daemon.
 * Subscribes to DBus signals for real-time state updates.
 *
 * States:
 * - idle: Indicator hidden
 * - recording: Red mic icon + last transcription text
 * - transcribing: Spinner icon + "Processing..."
 * - done: Check icon + "Copied!" (auto-hides after 5 seconds)
 */

import Clutter from "gi://Clutter";
import GLib from "gi://GLib";
import Gio from "gi://Gio";
import GObject from "gi://GObject";
import St from "gi://St";

import { Extension } from "resource:///org/gnome/shell/extensions/extension.js";
import * as Main from "resource:///org/gnome/shell/ui/main.js";
import * as PanelMenu from "resource:///org/gnome/shell/ui/panelMenu.js";

// DBus configuration - must match daemon
const DBUS_NAME = "com.github.frederikb.Voxscribe";
const DBUS_PATH = "/com/github/frederikb/Voxscribe";
const DBUS_INTERFACE = "com.github.frederikb.Voxscribe";

// Icon names for each state
const ICONS = {
  recording: "media-record-symbolic",
  transcribing: "emblem-synchronizing-symbolic",
  done: "emblem-ok-symbolic",
  partial: "dialog-warning-symbolic",
};

const VoxscribeIndicator = GObject.registerClass(
  class VoxscribeIndicator extends PanelMenu.Button {
    _init() {
      super._init(0.0, "Voxscribe Indicator", false);

      // Container box for icon + label
      this._box = new St.BoxLayout({
        style_class: "panel-status-menu-box",
      });
      this.add_child(this._box);

      // Icon
      this._icon = new St.Icon({
        icon_name: ICONS.recording,
        style_class: "system-status-icon",
      });
      this._box.add_child(this._icon);

      // Label for transcription preview
      this._label = new St.Label({
        text: "",
        y_align: Clutter.ActorAlign.CENTER,
        style_class: "voxscribe-label",
      });
      this._box.add_child(this._label);

      // Start hidden
      this.hide();

      // State tracking
      this._state = "idle";
      this._hideTimeoutId = null;
      this._dbusSignalId = null;
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

      if (state === "idle") {
        this.hide();
        return;
      }

      // Show indicator
      this.show();

      // Update icon
      if (ICONS[state]) {
        this._icon.set_icon_name(ICONS[state]);
      }

      // Update label based on state
      switch (state) {
        case "recording":
          // Show last ~15 chars of transcription
          if (text && text.length > 0) {
            const preview = text.length > 15 ? "..." + text.slice(-12) : text;
            this._label.set_text(preview);
          } else {
            this._label.set_text("Recording...");
          }
          break;

        case "transcribing":
          this._label.set_text("Processing...");
          break;

        case "done":
          this._label.set_text("Copied!");
          // Auto-hide after 5 seconds (only if still in done state)
          this._hideTimeoutId = GLib.timeout_add_seconds(
            GLib.PRIORITY_DEFAULT,
            5,
            () => {
              if (this._state === "done") {
                this.hide();
              }
              this._hideTimeoutId = null;
              return GLib.SOURCE_REMOVE;
            }
          );
          break;

        case "partial":
          this._label.set_text("Partial!");
          // Auto-hide after 5 seconds (only if still in partial state)
          this._hideTimeoutId = GLib.timeout_add_seconds(
            GLib.PRIORITY_DEFAULT,
            5,
            () => {
              if (this._state === "partial") {
                this.hide();
              }
              this._hideTimeoutId = null;
              return GLib.SOURCE_REMOVE;
            }
          );
          break;
      }
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
      super.destroy();
    }
  }
);

export default class VoxscribeExtension extends Extension {
  enable() {
    this._indicator = new VoxscribeIndicator();
    Main.panel.addToStatusArea(this.uuid, this._indicator);
    this._indicator.connectDbus();
    log("[Voxscribe] Extension enabled");
  }

  disable() {
    if (this._indicator) {
      this._indicator.destroy();
      this._indicator = null;
    }
    log("[Voxscribe] Extension disabled");
  }
}
