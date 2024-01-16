namespace ClassLibrary;

public class NormalRandom : Random
{
    // сохранённое предыдущее значение
    private double _prevSample = double.NaN;
    protected override double Sample()
    {
        // есть предыдущее значение? возвращаем его
        if (!double.IsNaN(_prevSample))
        {
            var result = _prevSample;
            _prevSample = double.NaN;
            return result;
        }

        // нет? вычисляем следующие два
        // Marsaglia polar method из википедии
        double u, v, s;
        do
        {
            u = 2 * base.Sample() - 1;
            v = 2 * base.Sample() - 1; // [-1, 1)
            s = u * u + v * v;
        }
        while (u <= -1 || v <= -1 || s >= 1 || s == 0);
        var r = Math.Sqrt(-2 * Math.Log(s) / s);

        _prevSample = r * v;
        return r * u;
    }
}
