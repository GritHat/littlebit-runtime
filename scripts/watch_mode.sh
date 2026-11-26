#!/usr/bin/env bash
# Switch a Wear OS watch between its normal state and one that leaves as much memory as possible to a model.
#
#   scripts/watch_mode.sh inference   # disable the background apps listed below, drop cached processes, keep the screen on
#   scripts/watch_mode.sh inference --aggressive   # also messages, timer, music, health, alarms, fall detection/SOS
#   scripts/watch_mode.sh inference --for 30       # the watch undoes it by itself after 30 minutes, connected or not
#   scripts/watch_mode.sh watch       # put everything back as it was, and check it
#   scripts/watch_mode.sh status
#
# Only apps that were enabled are disabled, and their previous state is written to the watch before anything changes,
# so "watch" restores it from any computer, also after a reboot or a lost connection. No data is deleted. While in
# inference mode the listed apps do not run: no WhatsApp notifications, no assistant, no calendar reminders.
# Messages, timer, music, health tracking, alarms and fall detection are only disabled with --aggressive; "watch" then also starts them again
# and checks that the services that ran before are running, and says so if a reboot is needed to be sure. The watch
# face and system services are never touched: disabling Google Play services and the like took the watch off Wi-Fi.
# Select the watch with ANDROID_SERIAL; ADB names the adb binary; WATCH_MODE_APPS replaces the list.
set -euo pipefail
ADB=${ADB:-adb}
STATE=/data/local/tmp/littlebit-watch-mode.state
APPS=${WATCH_MODE_APPS:-"com.whatsapp com.google.android.wearable.assistant com.samsung.android.bixby.agent
com.samsung.android.app.reminder com.samsung.android.calendar com.samsung.android.watch.stopwatch
com.sec.android.easyMover com.samsung.android.wear.voicerecorder com.samsung.android.messagebackuprestore
com.samsung.android.gallery.watch com.samsung.wear.contacts.sync com.samsung.android.watch.batterytile
com.samsung.android.watch.stf com.samsung.android.batterysavingsettings"}
VITAL=${WATCH_MODE_VITAL:-"com.samsung.android.wear.shealth com.google.android.wearable.healthservices
com.samsung.android.watch.alarm com.samsung.android.watch.safety_assistance"}
MORE=${WATCH_MODE_MORE:-"com.samsung.android.messaging com.samsung.android.watch.timer com.samsung.android.wearable.music
com.samsung.android.wear.musictransfer com.samsung.android.wear.blockednumber"}

sh_() { "$ADB" shell "$1" < /dev/null | tr -d '\r'; }
die() { echo "watch_mode: $*" >&2; exit 1; }
avail() { sh_ "grep MemAvailable /proc/meminfo" | awk '{printf "%d MB", $2 / 1024}'; }
enabled_state() { sh_ "dumpsys package $1 2> /dev/null | grep -m1 -oE 'enabled=[0-9]'" | cut -d= -f2; }
services() { sh_ "dumpsys activity services $1 2> /dev/null" | grep -oE 'ServiceRecord\{[^}]*\}' | grep -oE '[^ ]+/[^ }]+' | sort -u; }
alarms() { sh_ "dumpsys alarm" | grep -cE "^ +[A-Z_]+ #[0-9]+: Alarm\{[^}]* $1\}" || true; }
have_state() { [ "$(sh_ "[ -f $STATE ] && echo yes")" = yes ]; }

[ "$("$ADB" get-state 2> /dev/null)" = device ] || die "no device; connect the watch and set ANDROID_SERIAL"
case ${1:-} in
inference)
    have_state && die "already in inference mode; run '$0 watch' first"
    echo "available before: $(avail)"
    lines="stay_on $(sh_ 'settings get global stay_on_while_plugged_in')"
    aggressive=0; minutes=0
    shift
    while [ $# -gt 0 ]; do
        case $1 in
            --aggressive) aggressive=1 ;;
            --for) minutes=${2:?--for needs a number of minutes}; shift ;;
            *) die "unknown option $1" ;;
        esac
        shift
    done
    if [ $aggressive = 1 ]; then
        for p in $VITAL; do # remember what was running, to check it after the restore
            [ -n "$(sh_ "pidof $p")" ] && lines+=$'\n'"run $p"
            for c in $(services "$p"); do lines+=$'\n'"svc $p $c"; done
        done
        APPS="$APPS $MORE $VITAL"
        echo "aggressive: no messages, timers, music, health tracking, alarms or fall detection/SOS until '$0 watch'"
    fi
    for p in $APPS; do
        s=$(enabled_state "$p")
        case $s in 0 | 1) lines+=$'\n'"app $p $s"$'\n'"alarms $p $(alarms "$p")" ;; esac # else not installed or already disabled
    done
    printf '%s\n' "$lines" | "$ADB" shell "cat > $STATE.tmp && mv $STATE.tmp $STATE"
    [ "$(sh_ "wc -l < $STATE")" = "$(printf '%s\n' "$lines" | wc -l)" ] || die "could not write $STATE; nothing was changed"
    printf '%s\n' "$lines" | awk '$1 == "app" { print $2 }' | while read -r p; do
        sh_ "pm disable-user --user 0 $p" > /dev/null && echo "  disabled $p"
    done
    sh_ "am kill-all; settings put global stay_on_while_plugged_in 7; input keyevent KEYCODE_WAKEUP"
    if [ "$minutes" -gt 0 ]; then # the watch re-enables the apps by itself, connected or not; '$0 watch' still verifies
        en=$(printf '%s\n' "$lines" | awk '$1 == "app" { printf "pm enable --user 0 %s; ", $2 }')
        so=$(printf '%s\n' "$lines" | awk '$1 == "stay_on" { print $2 }')
        "$ADB" shell "(setsid nohup sh -c 'sleep $((minutes * 60)); [ -f $STATE ] || exit 0; $en settings put global stay_on_while_plugged_in $so' > /dev/null 2>&1 < /dev/null &)" < /dev/null
        echo "the watch re-enables these apps by itself in $minutes minutes"
    fi
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
    saved=$(sh_ "cat $STATE")
    sh_ "pkill -f 'sleep [0-9]+; \[ -f $STATE' 2> /dev/null" > /dev/null || true
    lost=""
    sleep 5
    while read -r _ p n; do # disabling an app cancels its scheduled alarms; most apps set them again when they start
        [ "$(alarms "$p")" -ge "$n" ] && continue
        sh_ "monkey -p $p -c android.intent.category.LAUNCHER 1" > /dev/null 2>&1 || true
        sleep 4
        sh_ "input keyevent KEYCODE_HOME"
        [ "$(alarms "$p")" -ge "$n" ] || lost+=" $p"
    done < <(printf '%s\n' "$saved" | awk '$1 == "alarms" && $3 > 0')
    vital=$(printf '%s\n' "$saved" | awk '$1 == "run" { print $2 }')
    if [ -n "$vital" ]; then
        missing() { # services and processes that ran before and do not now
            for p in $vital; do
                now=$(services "$p")
                had=$(printf '%s\n' "$saved" | awk -v p="$p" '$1 == "svc" && $2 == p' | wc -l)
                [ "$had" = 0 ] || [ -n "$(sh_ "pidof $p")" ] || echo "$p (process)"
                printf '%s\n' "$saved" | awk -v p="$p" '$1 == "svc" && $2 == p { print $3 }' | while read -r c; do
                    printf '%s\n' "$now" | grep -qxF "$c" || echo "$c"
                done
            done
        }
        sleep 10
        if [ -n "$(missing)" ]; then # not back by themselves: open each app once, which starts its services
            for p in $vital; do
                [ -n "$(sh_ "pidof $p")" ] || sh_ "monkey -p $p -c android.intent.category.LAUNCHER 1" > /dev/null 2>&1 || true
                sleep 3
            done
            sh_ "input keyevent KEYCODE_HOME"
            sleep 10
        fi
        m=$(missing)
        if [ -n "$m" ]; then
            [ -z "$lost" ] || echo "scheduled alarms or reminders not set again by:$lost"
            echo "apps are enabled again, but these were running before and are not now:"
            printf '%s\n' "$m" | sed 's/^/  /'
            echo "REBOOT THE WATCH to be sure that health tracking, alarms and fall detection/SOS work."
            sh_ "rm -f $STATE"
            exit 2
        fi
        echo "  health services are running as before"
    fi
    if [ -n "$lost" ]; then
        echo "apps are enabled again, but these have fewer scheduled alarms or reminders than before:$lost"
        echo "Open them once or REBOOT THE WATCH so that they are set again."
        sh_ "rm -f $STATE"
        exit 2
    fi
    sh_ "rm -f $STATE"
    echo "watch mode restored; stay_on=$(sh_ 'settings get global stay_on_while_plugged_in'), available: $(avail)"
    ;;
status)
    if have_state; then echo "inference mode; saved state:"; sh_ "cat $STATE" | sed 's/^/  /'; else echo "watch mode"; fi
    echo "available: $(avail); disabled packages: $(sh_ 'pm list packages -d' | wc -l)"
    ;;
*) die "usage: $0 inference [--aggressive] [--for MINUTES]|watch|status" ;;
esac
