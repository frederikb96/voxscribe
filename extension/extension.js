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
 *
 * Also owns the clipboard on the daemon's behalf: the daemon calls Copy() over DBus and GNOME
 * Shell holds the text itself, so no external process or X11 transfer is involved.
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

// DBus service exported by this extension - must match daemon's clipboard module
const SHELL_NAME = "com.github.frederikb.Voxscribe.Shell";
const SHELL_PATH = "/com/github/frederikb/Voxscribe/Shell";
const SHELL_IFACE_XML = `
<node>
  <interface name="${SHELL_NAME}">
    <method name="Copy">
      <arg type="s" direction="in" name="text"/>
      <arg type="b" direction="out" name="ok"/>
    </method>
  </interface>
</node>`;

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

/**
 * DBus service letting the daemon hand text to the Shell's own clipboard.
 */
class ShellClipboardService {
  constructor() {
    this._impl = Gio.DBusExportedObject.wrapJSObject(SHELL_IFACE_XML, this);
    this._impl.export(Gio.DBus.session, SHELL_PATH);
    this._nameId = Gio.bus_own_name_on_connection(
      Gio.DBus.session,
      SHELL_NAME,
      Gio.BusNameOwnerFlags.NONE,
      null,
      null
    );
  }

  Copy(text) {
    St.Clipboard.get_default().set_text(St.ClipboardType.CLIPBOARD, text);
    log(`[Voxscribe] Clipboard set: ${text.length} chars`);
    return true;
  }

  destroy() {
    if (this._nameId) {
      Gio.bus_unown_name(this._nameId);
      this._nameId = 0;
    }
    this._impl.unexport();
    this._impl = null;
  }
}

/**
 * Wrapping label whose minimum height is its full wrapped height.
 *
 * St.Viewport (the base of St.BoxLayout) allocates its content MAX(minimum, visible) and scrolls
 * over that range only. A wrapping St.Label reports a one-line minimum, so inside a scroll view
 * it would be squeezed to the visible height and ellipsized instead of scrolled.
 */
const FullHeightLabel = GObject.registerClass(
  class FullHeightLabel extends St.Label {
    vfunc_get_preferred_height(forWidth) {
      const [, natural] = super.vfunc_get_preferred_height(forWidth);
      return [natural, natural];
    }
  }
);

const VoxscribeIndicator = GObject.registerClass(
  class VoxscribeIndicator extends PanelMenu.Button {
    _init(settings) {
      super._init(0.0, "Voxscribe Indicator", false);

      this._settings = settings;
      this._state = "idle";
      // Popup follows the newest text while the reader sits at the end; scrolling up releases it,
      // returning to the very end re-arms it.
      this._followEnd = true;
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
     * Truncate text from START, showing the END (most recent speech).
     * Returns "...last part of text" format.
     */
    _truncateStart(text) {
      const maxWidth = this._settings.get_int("label-max-width");
      // Rough estimate: ~7px per char at 11px font
      const maxChars = Math.floor(maxWidth / 7);

      if (text.length <= maxChars) {
        return text;
      }
      return "..." + text.slice(-(maxChars - 3));
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

      this._textLabel = new FullHeightLabel({
        text: "No transcription yet",
        style_class: "voxscribe-popup-text",
        x_expand: true,
      });
      this._textLabel.clutter_text.set_line_wrap(true);
      this._textLabel.clutter_text.set_line_wrap_mode(0); // WORD
      this._textLabel.clutter_text.set_selectable(true);

      // BoxLayout is the scrollable viewport and constrains the wrap width
      const textBox = new St.BoxLayout({
        vertical: true,
        x_expand: true,
        style: "max-width: 400px;",
      });
      textBox.add_child(this._textLabel);
      this._scrollView.set_child(textBox);
      this._textItem.add_child(this._scrollView);
      this.menu.addMenuItem(this._textItem);

      const vadjustment = this._scrollView.vadjustment;
      vadjustment.connect("notify::value", (adj) => {
        this._followEnd = adj.value >= adj.upper - adj.page_size - 1;
      });
      vadjustment.connect("notify::upper", (adj) => {
        if (this._followEnd) {
          adj.value = adj.upper - adj.page_size;
        }
      });

      // Separator
      this.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

      // Copy button
      this._copyItem = new PopupMenu.PopupMenuItem("Copy to Clipboard");
      this._copyItem.connect("activate", () => this._copyToClipboard());
      this.menu.addMenuItem(this._copyItem);

      // Refresh button (recovery from frozen UI state)
      this._refreshItem = new PopupMenu.PopupMenuItem("Refresh");
      this._refreshItem.connect("activate", () => this.fetchInitialStatus());
      this.menu.addMenuItem(this._refreshItem);

      // Opening lands on the newest text
      this.menu.connect("open-state-changed", (_menu, isOpen) => {
        if (isOpen && this._state !== "idle") {
          this._followEnd = true;
          this._refreshPopup();
        }
      });
    }

    /**
     * Ask the daemon to re-deliver its current transcription to the clipboard.
     * The daemon owns the payload format and verifies the copy.
     */
    _copyToClipboard() {
      Gio.DBus.session.call(
        DBUS_NAME,
        DBUS_PATH,
        DBUS_INTERFACE,
        "CopyToClipboard",
        null,
        new GLib.VariantType("(b)"),
        Gio.DBusCallFlags.NONE,
        15000,
        null,
        (connection, result) => {
          let ok = false;
          try {
            [ok] = connection.call_finish(result).deepUnpack();
          } catch (e) {
            log(`[Voxscribe] Copy request failed: ${e.message}`);
          }
          this._flashLabel(ok ? "Copied!" : "Copy failed!");
        }
      );
    }

    /**
     * Show a short status in the panel; hides again when idle.
     */
    _flashLabel(text) {
      this._label.set_text(text);
      this.show();
      this._clearHideTimeout();
      if (this._state === "idle") {
        this._scheduleHide(3, "idle");
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
     * Ask the daemon for its state and full text; silent when it is not running.
     */
    _getStatus(onStatus) {
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
            const [state, text] = connection.call_finish(result).deepUnpack();
            onStatus(state, text);
          } catch (e) {
            log(`[Voxscribe] Could not fetch status: ${e.message}`);
          }
        }
      );
    }

    /**
     * Sync indicator with the daemon (recording may already be active).
     */
    fetchInitialStatus() {
      this._getStatus((state, text) => {
        log(`[Voxscribe] Initial status: ${state}`);
        this._updateState(state, text);
      });
    }

    /**
     * Show the daemon's full text in the popup.
     */
    _refreshPopup() {
      this._getStatus((_state, text) => {
        if (text.length > 0) {
          this._textLabel.set_text(text);
        }
      });
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
          // Show END of text in panel (most recent speech)
          this._label.set_text(this._truncateStart(text));
        } else {
          // New recording started - clear stale text
          this._textLabel.set_text("Recording...");
          this._label.set_text("Recording...");
        }
      } else if (state === "transcribing") {
        this._label.set_text("Processing...");
      } else if (AUTO_HIDE_STATES[state]) {
        const config = AUTO_HIDE_STATES[state];
        this._label.set_text(config.label);
        this._scheduleHide(config.seconds, state);
      }

      // The signal carries only the panel tail; an open popup wants the whole text
      if (this.menu.isOpen) {
        this._refreshPopup();
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
    this._clipboardService = new ShellClipboardService();
    log("[Voxscribe] Extension enabled");
  }

  disable() {
    if (this._clipboardService) {
      this._clipboardService.destroy();
      this._clipboardService = null;
    }
    if (this._indicator) {
      this._indicator.destroy();
      this._indicator = null;
    }
    this._settings = null;
    log("[Voxscribe] Extension disabled");
  }
}
