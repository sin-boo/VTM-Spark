// run.exe - windowless launcher (compiled /target:winexe by build-run.ps1).
// It never opens a console on its own: leftover cleanup, a stale-UI rebuild
// and pythonw -m backend run hidden, and it exits once the desk's splash
// window is up. When a start genuinely fails, a native dialog shows the reason
// and offers the repair console (start-menu.ps1 -Action run); the console
// only appears if the user says yes.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.NetworkInformation;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

internal static class Program
{
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool AllowSetForegroundWindow(int pid);
    private const int ASFW_ANY = -1;
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int MessageBoxW(IntPtr hWnd, string text, string caption, uint type);

    private delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    [DllImport("user32.dll")] private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll")] private static extern bool IsWindowVisible(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern IntPtr GetWindow(IntPtr hWnd, uint cmd);
    [DllImport("user32.dll")] private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint pid);
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    private static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int max);
    private const uint GW_OWNER = 4;

    private const string DeskTitle = "VTM Noble";
    private const string DeskMutex = "Local\\VTMNobleSingleInstance";
    // Held by run.exe while it launches, before the desk takes DeskMutex.
    private const string LaunchMutex = "Local\\VTMNobleLauncher";
    private const int DeskPort = 8765;

    private const uint MB_YESNO = 0x00000004;
    private const uint MB_ICONWARNING = 0x00000030;
    private const uint MB_SETFOREGROUND = 0x00010000;
    private const uint MB_TOPMOST = 0x00040000;
    private const int IDYES = 6;

    private static string Root = "";

    [STAThread]
    private static int Main()
    {
        string here = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        if (string.IsNullOrEmpty(here))
        {
            here = Environment.CurrentDirectory;
        }
        Root = here;
        Directory.SetCurrentDirectory(here);

        bool first;
        using (Mutex launching = new Mutex(true, LaunchMutex, out first))
        {
            if (!first)
            {
                // A click moments ago is still starting the desk.
                FocusDesk(TimeSpan.FromSeconds(3));
                return 0;
            }
            return Launch(here);
        }
    }

    private static int Launch(string here)
    {
        if (DeskRunning())
        {
            // Second click: bring the open (or still booting) desk forward.
            if (FocusDesk(TimeSpan.FromSeconds(3)))
            {
                return 0;
            }
            // No window: usually a desk that was just closed and is still
            // freeing GPU memory. Let it finish before calling it a hidden copy.
            if (!WaitDeskExit(TimeSpan.FromSeconds(10)))
            {
                if (FocusDeskOnce())
                {
                    return 0;
                }
                Log("single-instance lock held but no desk window found");
                if (!Ask("VTM Noble looks like it is already running, but its window cannot be found.\n\n"
                    + "Clean up leftover processes and start it again?"))
                {
                    return 0;
                }
                KillOrphans(false);
            }
        }

        string pyw = DeskPython(here);
        if (pyw == null)
        {
            Log("no usable .venv-build python");
            return Fail("No working Python environment (.venv-build).\nRun install.bat first.");
        }

        string failure;
        try
        {
            failure = Boot(pyw);
        }
        catch (Exception ex)
        {
            Log("start threw: " + ex.Message);
            failure = "Could not start the desk:\n" + ex.Message;
        }
        return failure == null ? 0 : Fail(failure);
    }

    // Everything between the click and the desk's splash. Null = started.
    private static string Boot(string pyw)
    {
        if (UiStale(Root) && !BuildUi())
        {
            return "The desk UI is not built and the rebuild failed.\nRun install.bat first.";
        }

        // Same sweep start-menu.ps1 does before a start: a leftover backend or
        // ui/_mock_boot.py on port 8765 would otherwise refuse the desk. The
        // script costs ~1.5 s, so it only runs when something could be left over.
        if (MaybeLeftovers())
        {
            KillOrphans(true);
        }
        return StartDesk(pyw);
    }

    // Cheap check for anything kill-orphans.ps1 -Fast could stop.
    private static bool MaybeLeftovers()
    {
        foreach (string name in new[] { "python", "pythonw", "VTMNoble", "RealStream" })
        {
            Process[] found = Process.GetProcessesByName(name);
            bool any = found.Length > 0;
            foreach (Process proc in found)
            {
                proc.Dispose();
            }
            if (any)
            {
                return true;
            }
        }
        try
        {
            foreach (IPEndPoint ep in IPGlobalProperties.GetIPGlobalProperties().GetActiveTcpListeners())
            {
                if (ep.Port == DeskPort)
                {
                    return true;
                }
            }
        }
        catch (Exception)
        {
            return true;
        }
        return false;
    }

    // Null = the desk's window is up; otherwise why it is not.
    private static string StartDesk(string pyw)
    {
        Process p;
        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pyw,
                Arguments = "-m backend --ui webview",
                WorkingDirectory = Root,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.EnvironmentVariables["PYTHONPATH"] = Root;
            p = Process.Start(psi);
        }
        catch (Exception ex)
        {
            Log("could not start pythonw: " + ex.Message);
            return "Could not start the desk:\n" + ex.Message;
        }
        if (p == null)
        {
            return "Could not start the desk.";
        }
        // The click made this process the foreground one; pass that on so the
        // desk (a grandchild, behind the venv launcher) may come to the front.
        AllowSetForegroundWindow(ASFW_ANY);

        // Stay until the desk's splash is up so a failed start still gets
        // reported. The venv's pythonw.exe is only a launcher: the real
        // interpreter is its child and owns the window, so look for the window
        // itself; the launcher's exit code is still the desk's.
        DateTime deadline = DateTime.UtcNow.AddSeconds(60);
        while (DateTime.UtcNow < deadline)
        {
            if (p.WaitForExit(150))
            {
                if (p.ExitCode == 0)
                {
                    return null;
                }
                Log("desk exited with code " + p.ExitCode + " before its window opened");
                return "The desk exited before its window opened (code " + p.ExitCode + ").";
            }
            if (FindDeskWindow(true) != IntPtr.Zero)
            {
                return null;
            }
        }
        Log("desk window did not appear within 60 s");
        return null;
    }

    // Rebuild ui/dist hidden. True = ui/dist is usable (fresh, or the last
    // build when the rebuild could not run).
    private static bool BuildUi()
    {
        string ui = Path.Combine(Root, "ui");
        string index = Path.Combine(ui, "dist", "index.html");
        Log("ui source newer than ui\\dist - rebuilding hidden");
        int code = RunHidden("cmd.exe", "/c npm run build", ui, TimeSpan.FromMinutes(10));
        if (code == 0)
        {
            Log("ui rebuilt");
            return true;
        }
        Log("ui rebuild failed (exit " + code + ")" + (File.Exists(index) ? "; launching last ui\\dist" : ""));
        return File.Exists(index);
    }

    private static void KillOrphans(bool fast)
    {
        string ps1 = Path.Combine(Root, "backend", "packaging", "kill-orphans.ps1");
        if (!File.Exists(ps1))
        {
            return;
        }
        string args = "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File \""
            + ps1 + "\" -Quiet" + (fast ? " -Fast" : "");
        RunHidden("powershell.exe", args, Root, TimeSpan.FromSeconds(fast ? 30 : 90));
    }

    // Exit code of a hidden child; -1 = could not start, -2 = timed out (killed).
    private static int RunHidden(string file, string args, string cwd, TimeSpan timeout)
    {
        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = file,
                Arguments = args,
                WorkingDirectory = cwd,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            psi.EnvironmentVariables["PYTHONPATH"] = Root;
            using (Process p = Process.Start(psi))
            {
                if (p == null)
                {
                    return -1;
                }
                if (p.WaitForExit((int)timeout.TotalMilliseconds))
                {
                    return p.ExitCode;
                }
                try { p.Kill(); } catch (Exception) { }
                return -2;
            }
        }
        catch (Exception ex)
        {
            Log("could not run " + file + ": " + ex.Message);
            return -1;
        }
    }

    // Report a failed start in a dialog. Yes = open the repair console.
    private static int Fail(string reason)
    {
        StringBuilder sb = new StringBuilder(reason);
        string tail = LogTail(8);
        if (tail.Length > 0)
        {
            sb.Append("\n\nLast log lines:\n").Append(tail);
        }
        sb.Append("\n\nOpen the repair console?");
        if (Ask(sb.ToString()))
        {
            return RunMenu(Root);
        }
        return 1;
    }

    private static bool Ask(string text)
    {
        return MessageBoxW(IntPtr.Zero, text, DeskTitle,
            MB_YESNO | MB_ICONWARNING | MB_SETFOREGROUND | MB_TOPMOST) == IDYES;
    }

    private static string LogPath()
    {
        return Path.Combine(Root, "models", "vtm_noble.log");
    }

    private static void Log(string msg)
    {
        try
        {
            string path = LogPath();
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.AppendAllText(path, "[" + DateTime.Now.ToString("HH:mm:ss") + "] run.exe: " + msg + "\n");
        }
        catch (Exception) { }
    }

    private static string LogTail(int count)
    {
        try
        {
            string path = LogPath();
            if (!File.Exists(path))
            {
                return "";
            }
            string[] lines = File.ReadAllLines(path);
            List<string> keep = new List<string>();
            for (int i = Math.Max(0, lines.Length - count); i < lines.Length; i++)
            {
                string line = lines[i].TrimEnd();
                if (line.Length > 110)
                {
                    line = line.Substring(0, 107) + "...";
                }
                keep.Add(line);
            }
            return string.Join("\n", keep.ToArray());
        }
        catch (Exception)
        {
            return "";
        }
    }

    private static bool DeskRunning()
    {
        Mutex m;
        if (!Mutex.TryOpenExisting(DeskMutex, out m))
        {
            return false;
        }
        m.Dispose();
        return true;
    }

    // True once the desk's single-instance lock is gone (it finished exiting).
    private static bool WaitDeskExit(TimeSpan patience)
    {
        DateTime deadline = DateTime.UtcNow + patience;
        while (DeskRunning())
        {
            if (DateTime.UtcNow >= deadline)
            {
                return false;
            }
            Thread.Sleep(250);
        }
        return true;
    }

    private static bool FocusDesk(TimeSpan patience)
    {
        DateTime deadline = DateTime.UtcNow + patience;
        while (true)
        {
            if (FocusDeskOnce())
            {
                return true;
            }
            if (DateTime.UtcNow >= deadline)
            {
                return false;
            }
            Thread.Sleep(250);
        }
    }

    // The desk (or its splash, mid-launch) comes forward.
    private static bool FocusDeskOnce()
    {
        IntPtr hwnd = FindDeskWindow(false);
        if (hwnd == IntPtr.Zero)
        {
            return false;
        }
        if (IsIconic(hwnd))
        {
            ShowWindow(hwnd, 9); // SW_RESTORE
        }
        SetForegroundWindow(hwnd);
        return true;
    }

    // A visible top-level "VTM Noble" window from another process. With
    // pythonOnly, only the desk's own windows count.
    private static IntPtr FindDeskWindow(bool pythonOnly)
    {
        uint self = (uint)Process.GetCurrentProcess().Id;
        IntPtr found = IntPtr.Zero;
        StringBuilder title = new StringBuilder(64);
        EnumWindows(delegate (IntPtr hwnd, IntPtr unused)
        {
            if (!IsWindowVisible(hwnd) || GetWindow(hwnd, GW_OWNER) != IntPtr.Zero)
            {
                return true;
            }
            title.Length = 0;
            GetWindowText(hwnd, title, title.Capacity);
            if (title.ToString() != DeskTitle)
            {
                return true;
            }
            uint pid;
            GetWindowThreadProcessId(hwnd, out pid);
            if (pid == self)
            {
                return true;
            }
            if (pythonOnly && !IsPython(pid))
            {
                return true;
            }
            found = hwnd;
            return false;
        }, IntPtr.Zero);
        return found;
    }

    private static bool IsPython(uint pid)
    {
        try
        {
            using (Process proc = Process.GetProcessById((int)pid))
            {
                string name = proc.ProcessName.ToLowerInvariant();
                return name == "pythonw" || name == "python";
            }
        }
        catch (Exception)
        {
            return false;
        }
    }

    // Same test as Resolve-VtmDeskPython: pythonw is only usable while the
    // base interpreter named in pyvenv.cfg still exists.
    private static string DeskPython(string root)
    {
        string venv = Path.Combine(root, ".venv-build");
        string pyw = Path.Combine(venv, "Scripts", "pythonw.exe");
        string cfg = Path.Combine(venv, "pyvenv.cfg");
        if (!File.Exists(pyw) || !File.Exists(cfg))
        {
            return null;
        }
        foreach (string line in File.ReadAllLines(cfg))
        {
            int eq = line.IndexOf('=');
            if (eq < 0 || line.Substring(0, eq).Trim() != "home")
            {
                continue;
            }
            string home = line.Substring(eq + 1).Trim();
            bool ok = File.Exists(Path.Combine(home, "pythonw.exe"))
                && Directory.Exists(Path.Combine(home, "Lib", "encodings"));
            return ok ? pyw : null;
        }
        return null;
    }

    // Mirrors Test-UiStale in start-menu.ps1.
    private static bool UiStale(string root)
    {
        string index = Path.Combine(root, "ui", "dist", "index.html");
        if (!File.Exists(index))
        {
            return true;
        }
        DateTime built = File.GetLastWriteTimeUtc(index);
        string ui = Path.Combine(root, "ui");
        string html = Path.Combine(ui, "index.html");
        if (File.Exists(html) && File.GetLastWriteTimeUtc(html) > built)
        {
            return true;
        }
        foreach (string dir in new[] { "src", "public" })
        {
            string path = Path.Combine(ui, dir);
            if (!Directory.Exists(path))
            {
                continue;
            }
            foreach (string f in Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories))
            {
                if (Path.GetFileName(f).StartsWith("_"))
                {
                    continue;
                }
                if (File.GetLastWriteTimeUtc(f) > built)
                {
                    return true;
                }
            }
        }
        return false;
    }

    // The only place a console appears, and only after the user said yes.
    private static int RunMenu(string here)
    {
        string ps1 = Path.Combine(here, "backend", "packaging", "start-menu.ps1");
        Process p = Process.Start(new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = "-NoProfile -ExecutionPolicy Bypass -File \"" + ps1 + "\" -Action run",
            WorkingDirectory = here,
            UseShellExecute = false,
        });
        return p == null ? 1 : 0;
    }
}
