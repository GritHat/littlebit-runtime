package io.github.grithat.lbsudoku;

import android.app.Activity;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Solves Sudoku puzzles on the watch with a Next-Concept model, through the littlebit command line tool. */
public class MainActivity extends Activity {
    static final String[][] MODELS = {
        {"easy-lb0.55.gguf", "LittleBit 0.55 bit", "6.8 MB"},
        {"easy-dense-q8.gguf", "dense 8 bit", "31 MB"},
    };

    final ExecutorService worker = Executors.newSingleThreadExecutor();
    final Handler ui = new Handler(Looper.getMainLooper());
    final List<String[]> puzzles = new ArrayList<>(); // {puzzle, solution}
    Process proc;
    Writer toProc;
    BufferedReader fromProc;
    int model = 0, index = 0, solvedCount = 0, triedCount = 0;
    boolean busy;
    String puzzle, answer, solution, status = "loading model...";
    long ms;
    Board board;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        board = new Board(this, this);
        setContentView(board);
        worker.execute(() -> {
            try {
                BufferedReader r = new BufferedReader(new InputStreamReader(getAssets().open("puzzles.txt")));
                for (String l; (l = r.readLine()) != null; ) {
                    String[] p = l.trim().split(" ");
                    if (p.length == 2) puzzles.add(p);
                }
                start(model);
                next();
            } catch (Exception e) {
                show("error: " + e.getMessage());
            }
        });
    }

    File asset(String name) throws Exception {
        File f = new File(getFilesDir(), name);
        if (f.exists()) return f;
        File tmp = new File(getFilesDir(), name + ".tmp");
        try (InputStream in = getAssets().open(name); OutputStream out = new FileOutputStream(tmp)) {
            byte[] buf = new byte[1 << 16];
            for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
        }
        if (!tmp.renameTo(f)) throw new Exception("cannot write " + f);
        return f;
    }

    void start(int m) throws Exception {
        if (proc != null) proc.destroy();
        show("loading " + MODELS[m][1] + "...");
        String exe = getApplicationInfo().nativeLibraryDir + "/liblittlebit_cli.so";
        proc = new ProcessBuilder(exe, "ncm", asset(MODELS[m][0]).getPath(), "-t", "1").redirectErrorStream(false).start();
        toProc = new OutputStreamWriter(proc.getOutputStream());
        fromProc = new BufferedReader(new InputStreamReader(proc.getInputStream()));
        InputStream err = proc.getErrorStream();
        new Thread(() -> { try { while (err.read() >= 0) { } } catch (Exception ignored) { } }).start();
        solvedCount = triedCount = 0;
    }

    void next() throws Exception {
        String[] p = puzzles.get(index++ % puzzles.size());
        puzzle = p[0];
        solution = p[1];
        answer = null;
        show("thinking...");
        long t = System.nanoTime();
        toProc.write(puzzle + "\n");
        toProc.flush();
        String a = fromProc.readLine();
        ms = (System.nanoTime() - t) / 1000000;
        if (a == null || a.length() != 81) throw new Exception("no answer from the model");
        answer = a;
        triedCount++;
        if (a.equals(solution)) solvedCount++;
        show(null);
    }

    void show(String s) {
        ui.post(() -> { status = s; board.invalidate(); });
    }

    void tap(boolean longPress) {
        if (busy) return;
        busy = true;
        worker.execute(() -> {
            try {
                if (longPress) {
                    model = 1 - model;
                    start(model);
                }
                next();
            } catch (Exception e) {
                show("error: " + e.getMessage());
            }
            busy = false;
        });
    }

    @Override
    protected void onDestroy() {
        if (proc != null) proc.destroy();
        worker.shutdownNow();
        super.onDestroy();
    }

    /** Static, because the d8 of build-tools 34 fails on inner classes compiled by recent javac. */
    static class Board extends View {
        final Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        final MainActivity a;
        long down;

        Board(Context c, MainActivity a) {
            super(c);
            this.a = a;
            setBackgroundColor(Color.BLACK);
        }

        @Override
        public boolean onTouchEvent(MotionEvent e) {
            if (e.getAction() == MotionEvent.ACTION_DOWN) down = e.getEventTime();
            if (e.getAction() == MotionEvent.ACTION_UP) a.tap(e.getEventTime() - down > 600);
            return true;
        }

        @Override
        protected void onDraw(Canvas c) {
            float w = getWidth(), h = getHeight(), side = Math.min(w, h) * 0.64f, cell = side / 9;
            float x0 = (w - side) / 2, y0 = (h - side) / 2 + h * 0.02f;
            p.setTextAlign(Paint.Align.CENTER);
            p.setColor(Color.rgb(255, 196, 0));
            p.setTextSize(h * 0.05f);
            p.setFakeBoldText(true);
            c.drawText(MODELS[a.model][1] + " · " + MODELS[a.model][2], w / 2, y0 - h * 0.035f, p);
            p.setFakeBoldText(false);
            for (int i = 0; i < 81 && a.puzzle != null; i++) {
                char given = a.puzzle.charAt(i);
                char d = given != '0' ? given : a.answer != null ? a.answer.charAt(i) : ' ';
                if (d == ' ') continue;
                boolean wrong = given == '0' && a.answer != null && d != a.solution.charAt(i);
                p.setColor(given != '0' ? Color.WHITE : wrong ? Color.rgb(255, 80, 80) : Color.rgb(80, 220, 255));
                p.setTextSize(cell * 0.72f);
                c.drawText(String.valueOf(d), x0 + (i % 9 + 0.5f) * cell, y0 + (i / 9 + 0.5f) * cell + cell * 0.26f, p);
            }
            p.setColor(Color.GRAY);
            for (int i = 0; i <= 9; i++) {
                p.setStrokeWidth(i % 3 == 0 ? 3 : 1);
                c.drawLine(x0 + i * cell, y0, x0 + i * cell, y0 + side, p);
                c.drawLine(x0, y0 + i * cell, x0 + side, y0 + i * cell, p);
            }
            p.setTextSize(h * 0.045f);
            String line = a.status != null ? a.status
                : (a.answer.equals(a.solution) ? "solved" : "wrong") + " in " + a.ms + " ms · " + a.solvedCount + "/" + a.triedCount;
            p.setColor(a.status != null ? Color.LTGRAY : a.answer.equals(a.solution) ? Color.rgb(120, 230, 120) : Color.rgb(255, 80, 80));
            c.drawText(line, w / 2, y0 + side + h * 0.065f, p);
        }
    }
}
