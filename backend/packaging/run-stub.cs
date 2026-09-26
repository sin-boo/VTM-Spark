// run.exe — windowless launcher (compiled /target:winexe by build-run.ps1).
// Healthy install: start pythonw -m backend directly so the splash is the
// first thing on screen. Anything that needs a repair (missing venv, stale UI,
// start failure) falls back to start-menu.ps1 in a visible console.
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Threading;

internal static class Program
{
    [DllImport("user32.dll")] private static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] private static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] private static extern bool IsIconic(IntPtr hWnd);

    private const string DeskTitle = "VTM Noble";
    private const string DeskMutex = "Local\\VTMNobleSingleInstance";

    [STAThread]
    private static int Main()
    {
        string here = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        if (string.IsNullOrEmpty(here))
        {
            here = Environment.CurrentDirectory;
        }
        Directory.SetCurrentDirectory(here);

        if (DeskRunning())
        {
            // Second click: bring the open desk forward instead of restarting it.
            if (FocusDesk())
            {
                return 0;
            }
            return RunMenu(here);
        }

        string pyw = DeskPython(here);
        if (pyw == null || UiStale(here))
        {
            return RunMenu(here);
        }

        Process p;
        try
        {
            ProcessStartInfo psi = new ProcessStartInfo
            {
                FileName = pyw,
                Arguments = "-m backend --ui webview",
                WorkingDirectory = here,
                UseShellExecute = false,
            };
            psi.EnvironmentVariables["PYTHONPATH"] = here;
            p = Process.Start(psi);
        }
        catch (Exception)
        {
            return RunMenu(here);
        }
        if (p == null)
        {
            return RunMenu(here);
        }

        // Stay (invisible) until the splash is up so a failed start still gets
        // the menu's leftover cleanup and log tail.
        DateTime deadline = DateTime.UtcNow.AddSeconds(60);
        while (DateTime.UtcNow < deadline)
        {
            if (p.WaitForExit(150))
            {
                return p.ExitCode == 0 ? 0 : RunMenu(here);
            }
            p.Refresh();
            if (p.MainWindowHandle != IntPtr.Zero)
            {
                return 0;
            }
        }
        return 0;
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

    private static bool FocusDesk()
    {
        foreach (string name in new[] { "pythonw", "python" })
        {
            foreach (Process proc in Process.GetProcessesByName(name))
            {
                IntPtr hwnd = proc.MainWindowHandle;
                if (hwnd == IntPtr.Zero || proc.MainWindowTitle != DeskTitle)
                {
                    continue;
                }
                if (IsIconic(hwnd))
                {
                    ShowWindow(hwnd, 9); // SW_RESTORE
                }
                SetForegroundWindow(hwnd);
                return true;
            }
        }
        return false;
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
