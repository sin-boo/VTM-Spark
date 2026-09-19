using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

internal static class Program
{
    private static int Main()
    {
        string here = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        if (string.IsNullOrEmpty(here))
        {
            here = Environment.CurrentDirectory;
        }
        Directory.SetCurrentDirectory(here);
        string ps1 = Path.Combine(here, "backend", "packaging", "start-menu.ps1");
        Process p = Process.Start(new ProcessStartInfo
        {
            FileName = "powershell.exe",
            Arguments = "-NoProfile -ExecutionPolicy Bypass -File \"" + ps1 + "\" -Action run",
            WorkingDirectory = here,
            UseShellExecute = false,
        });
        if (p == null)
        {
            return 1;
        }
        p.WaitForExit();
        return p.ExitCode;
    }
}
