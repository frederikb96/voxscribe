/**
 * Voxscribe Extension Preferences
 */

import Adw from "gi://Adw";
import Gio from "gi://Gio";
import Gtk from "gi://Gtk";

import {
  ExtensionPreferences,
} from "resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js";

export default class VoxscribePreferences extends ExtensionPreferences {
  fillPreferencesWindow(window) {
    const settings = this.getSettings();

    // Create preferences page
    const page = new Adw.PreferencesPage({
      title: "Voxscribe",
      icon_name: "audio-input-microphone-symbolic",
    });
    window.add(page);

    // Appearance group
    const appearanceGroup = new Adw.PreferencesGroup({
      title: "Appearance",
      description: "Configure how the indicator looks in the panel",
    });
    page.add(appearanceGroup);

    // Label width setting
    const widthRow = new Adw.SpinRow({
      title: "Label Width",
      subtitle: "Maximum width of the transcription preview (pixels)",
      adjustment: new Gtk.Adjustment({
        lower: 50,
        upper: 500,
        step_increment: 10,
        page_increment: 50,
        value: settings.get_int("label-max-width"),
      }),
    });
    appearanceGroup.add(widthRow);

    // Bind setting
    settings.bind(
      "label-max-width",
      widthRow,
      "value",
      Gio.SettingsBindFlags.DEFAULT
    );
  }
}
