#!/usr/bin/env bash
# Switch a Wear OS watch between its normal state and one that leaves as much memory as possible to a model.
#
#   scripts/watch_mode.sh inference   # disable the background apps listed below, drop cached processes, keep the screen on
#   scripts/watch_mode.sh watch       # put everything back as it was
#   scripts/watch_mode.sh status
#
# Only apps that were enabled are disabled, and their previous state is written to the watch before anything changes,
# so "watch" restores it from any computer, also after a reboot or a lost connection. No data is deleted. While in
# inference mode the listed apps do not run: no WhatsApp notifications, no assistant, no calendar reminders.
# Health tracking, alarms, fall detection, the watch face and system services are left alone.
# Select the watch with ANDROID_SERIAL; ADB names the adb binary; WATCH_MODE_APPS replaces the list.
set -euo pipefail
ADB=${ADB:-adb}
STATE=/data/local/tmp/littlebit-watch-mode.state
APPS=${WATCH_MODE_APPS:-"com.whatsapp com.google.android.wearable.assistant com.samsung.android.bixby.agent
com.samsung.android.app.reminder com.samsung.android.calendar com.samsung.android.watch.stopwatch
com.sec.android.easyMover com.samsung.android.wear.voicerecorder com.samsung.android.messagebackuprestore
com.samsung.android.gallery.watch com.samsung.wear.contacts.sync com.samsung.android.watch.batterytile
com.samsung.android.watch.stf com.samsung.android.batterysavingsettings"}

sh_() { "$ADB" shell "$1" < /dev/null | tr -d '\r'; }
die() { echo "watch_mode: $*" >&2; exit 1; }
avail() { sh_ "grep MemAvailable /proc/meminfo" | awk '{printf "%d MB", $2 / 1024}'; }
enabled_state() { sh_ "dumpsys package $1 2> /dev/null | grep -m1 -oE 'enabled=[0-9]'" | cut -d= -f2; }
have_state() { [ "$(sh_ "[ -f $STATE ] && echo yes")" = yes ]; }

[ "$("$ADB" get-state 2> /dev/null)" = device ] || die "no device; connect the watch and set ANDROID_SERIAL"
case ${1:-} in
inference)
    have_state && die "already in inference mode; run '$0 watch' first"
    echo "available before: $(avail)"
    lines="stay_on $(sh_ 'settings get global stay_on_while_plugged_in')"
    for p in $APPS; do
        s=$(enabled_state "$p")
        case $s in 0 | 1) lines+=$'\n'"app $p $s" ;; esac # not installed or already disabled: leave alone
    done
    printf '%s\n' "$lines" | "$ADB" shell "cat > $STATE.tmp && mv $STATE.tmp $STATE"
    [ "$(sh_ "wc -l < $STATE")" = "$(printf '%s\n' "$lines" | wc -l)" ] || die "could not write $STATE; nothing was changed"
    printf '%s\n' "$lines" | awk '$1 == "app" { print $2 }' | while read -r p; do
        sh_ "pm disable-user --user 0 $p" > /dev/null && echo "  disabled $p"
    done
    sh_ "am kill-all; settings put global stay_on_while_plugged_in 7; input keyevent KEYCODE_WAKEUP"
    sleep 3
    echo "available now:    $(avail)"
    ;;
watch)
    have_state || die "not in inference mode (no $STATE on the watch)"
    bad=0
    while read -r kind a b; do
        case $kind in
            stay_on) sh_ "settings put global stay_on_while_plugged_in $a" ;;
            app)
                if [ "$b" = 0 ]; then
                    sh_ "pm default-state --user 0 $a" > /dev/null 2>&1 || sh_ "pm enable --user 0 $a" > /dev/null
                else
                    sh_ "pm enable --user 0 $a" > /dev/null
                fi
                case $(enabled_state "$a") in 0 | 1) echo "  restored $a" ;; *) echo "  FAILED   $a"; bad=1 ;; esac
                ;;
        esac
    done < <(sh_ "cat $STATE")
    [ $bad = 0 ] || die "some apps were not restored; $STATE is kept, run '$0 watch' again"
    sh_ "rm -f $STATE"
    echo "watch mode restored; stay_on=$(sh_ 'settings get global stay_on_while_plugged_in'), available: $(avail)"
    ;;
status)
    if have_state; then echo "inference mode; saved state:"; sh_ "cat $STATE" | sed 's/^/  /'; else echo "watch mode"; fi
    echo "available: $(avail); disabled packages: $(sh_ 'pm list packages -d' | wc -l)"
    ;;
*) die "usage: $0 inference|watch|status" ;;
esac
