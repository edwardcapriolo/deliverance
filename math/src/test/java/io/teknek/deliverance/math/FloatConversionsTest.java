package io.teknek.deliverance.math;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for the {@code FloatConversions} utility class, which provides methods
 * for performing conversions and operations involving floating-point values in
 * different formats, such as BFloat16 and IEEE Float16.
 * <p>
 * These tests verify the accuracy of conversions, rounding behaviors, handling of
 * special values like zeros, infinities, and NaN, as well as arithmetic operations for
 * IEEE Float16 values.
 */
class FloatConversionsTest {

    @Test
    void convertsBFloat16RawBitsToFloat32() {
        assertEquals(1.0f, FloatConversions.bFloat16ToFloat32((short) 0x3f80));
        assertEquals(-2.0f, FloatConversions.bFloat16ToFloat32((short) 0xc000));
        assertEquals(Float.POSITIVE_INFINITY,
                FloatConversions.bFloat16ToFloat32((short) 0x7f80));
        assertEquals(Float.NEGATIVE_INFINITY,
                FloatConversions.bFloat16ToFloat32((short) 0xff80));
    }

    @Test
    void roundsFloat32ToNearestEvenBFloat16() {
        assertEquals((short) 0x3f80, FloatConversions.float32ToBFloat16(1.0f));
        assertEquals((short) 0x3f80,
                FloatConversions.float32ToBFloat16(Float.intBitsToFloat(0x3f808000)),
                "a midpoint rounds to the even lower mantissa");
        assertEquals((short) 0x3f82,
                FloatConversions.float32ToBFloat16(Float.intBitsToFloat(0x3f818000)),
                "a midpoint rounds to the even upper mantissa");
        assertEquals((short) 0x3f81,
                FloatConversions.float32ToBFloat16(Float.intBitsToFloat(0x3f808001)));
    }

    @Test
    void preservesBFloat16SignedZeroAndHandlesSpecialValues() {
        assertEquals((short) 0x0000, FloatConversions.float32ToBFloat16(0.0f));
        assertEquals((short) 0x8000, FloatConversions.float32ToBFloat16(-0.0f));
        assertEquals((short) 0x7f80, FloatConversions.float32ToBFloat16(Float.POSITIVE_INFINITY));
        assertEquals((short) 0xff80, FloatConversions.float32ToBFloat16(Float.NEGATIVE_INFINITY));
        assertEquals(FloatConversions.bFloat16NaN,
                FloatConversions.float32ToBFloat16(Float.NaN));
    }

    @Test
    void convertsAlternativeFloat16Values() {
        assertEquals(0.0f, FloatConversions.float16ToFloat32Alt((short) 0x0000));
        assertEquals(-0.0f, FloatConversions.float16ToFloat32Alt((short) 0x8000));
        assertEquals(1.0f, FloatConversions.float16ToFloat32Alt((short) 0x3c00));
        assertEquals(-2.0f, FloatConversions.float16ToFloat32Alt((short) 0xc000));
    }

    @Test
    void handlesIeeeFloat16SpecialValues() {
        assertEquals((short) 0x7c00,
                FloatConversions.addIeeeFloat16((short) 0x7c00, (short) 0x3c00));
        assertEquals((short) 0xfc00,
                FloatConversions.mulIeeeFloat16((short) 0xfc00, (short) 0x3c00));

        short nan = FloatConversions.addIeeeFloat16((short) 0x7c01, (short) 0x3c00);
        assertEquals((short) 0x7fff, nan);
        assertTrue(Float.isNaN(FloatConversions.bFloat16ToFloat32(FloatConversions.bFloat16NaN)));
        assertFalse(Float.isNaN(FloatConversions.bFloat16ToFloat32((short) 0x7f80)));
    }

    @Test
    public void testZerosAndOnes() {
        assertEquals((short) 0x0000, FloatConversions.float32ToBFloat16(0.0f));
        assertEquals((short) 0x8000, FloatConversions.float32ToBFloat16(-0.0f));
        assertEquals((short) 0x3f80, FloatConversions.float32ToBFloat16(1.0f));
        assertEquals((short) 0xbf80, FloatConversions.float32ToBFloat16(-1.0f));

        assertEquals(0.0f, FloatConversions.bFloat16ToFloat32((short) 0x0000));
        assertEquals(-0.0f, FloatConversions.bFloat16ToFloat32((short) 0x8000));
        assertEquals(1.0f, FloatConversions.bFloat16ToFloat32((short) 0x3f80));
        assertEquals(-1.0f, FloatConversions.bFloat16ToFloat32((short) 0xbf80));
    }

    @Test
    public void testInfinities() {
        assertEquals((short) 0x7f80, FloatConversions.float32ToBFloat16(Float.POSITIVE_INFINITY));
        assertEquals((short) 0xff80, FloatConversions.float32ToBFloat16(Float.NEGATIVE_INFINITY));

        assertEquals(Float.POSITIVE_INFINITY, FloatConversions.bFloat16ToFloat32((short) 0x7f80));
        assertEquals(Float.NEGATIVE_INFINITY, FloatConversions.bFloat16ToFloat32((short) 0xff80));
    }

    @Test
    public void testRoundToNearestEven() {
        // Discarded < 0.5 (round down)
        float f1 = Float.intBitsToFloat(0x3f807fff);
        assertEquals((short) 0x3f80, FloatConversions.float32ToBFloat16(f1));

        // Discarded > 0.5 (round up)
        float f2 = Float.intBitsToFloat(0x3f808001);
        assertEquals((short) 0x3f81, FloatConversions.float32ToBFloat16(f2));

        // Exact tie (0.5), LSB is even (0) -> round down (keep even)
        float f3 = Float.intBitsToFloat(0x3f808000);
        assertEquals((short) 0x3f80, FloatConversions.float32ToBFloat16(f3));

        // Exact tie (0.5), LSB is odd (1) -> round up (to even)
        float f4 = Float.intBitsToFloat(0x3f818000);
        assertEquals((short) 0x3f82, FloatConversions.float32ToBFloat16(f4));

        // Negative numbers: exact tie (0.5), LSB is even -> round down in magnitude
        float f5 = Float.intBitsToFloat(0xbf808000);
        assertEquals((short) 0xbf80, FloatConversions.float32ToBFloat16(f5));

        // Negative numbers: exact tie (0.5), LSB is odd -> round up in magnitude (to even)
        float f6 = Float.intBitsToFloat(0xbf818000);
        assertEquals((short) 0xbf82, FloatConversions.float32ToBFloat16(f6));
    }

    @Test
    public void testSubnormals() {
        // Subnormal rounding
        float subnormal1 = Float.intBitsToFloat(0x00008000); // exact tie, LSB is 0 -> 0
        assertEquals((short) 0x0000, FloatConversions.float32ToBFloat16(subnormal1));

        float subnormal2 = Float.intBitsToFloat(0x00018000); // exact tie, LSB is 1 -> 2
        assertEquals((short) 0x0002, FloatConversions.float32ToBFloat16(subnormal2));

        float subnormal3 = Float.intBitsToFloat(0x00008001); // > 0.5 -> 1
        assertEquals((short) 0x0001, FloatConversions.float32ToBFloat16(subnormal3));
    }

    @Test
    public void testMaxFiniteRounding() {
        // Float rounding to max finite bfloat16
        float maxFinite = Float.intBitsToFloat(0x7f7f7fff);
        assertEquals((short) 0x7f7f, FloatConversions.float32ToBFloat16(maxFinite));

        // Float rounding to infinity on tie with odd LSB
        float roundToInf = Float.intBitsToFloat(0x7f7f8000);
        assertEquals((short) 0x7f80, FloatConversions.float32ToBFloat16(roundToInf));
    }

    @Test
    void performsBasicIeeeFloat16Arithmetic() {
        assertEquals((short) 0x4000,
                FloatConversions.addIeeeFloat16((short) 0x3c00, (short) 0x3c00));
        assertEquals((short) 0x0000,
                FloatConversions.addIeeeFloat16((short) 0x3c00, (short) 0xbc00));
        assertEquals((short) 0x3800,
                FloatConversions.subIeeeFloat16((short) 0x3c00, (short) 0x3800));
        assertEquals((short) 0xb800,
                FloatConversions.subIeeeFloat16((short) 0x3800, (short) 0x3c00));
        assertEquals((short) 0x4200,
                FloatConversions.mulIeeeFloat16((short) 0x3e00, (short) 0x4000));
    }

    @Test
    void handlesIeeeFloat16ArithmeticBoundaries() {
        assertEquals((short) 0x7c00,
                FloatConversions.addIeeeFloat16((short) 0x7bff, (short) 0x7bff));
        assertEquals((short) 0x0000,
                FloatConversions.subIeeeFloat16((short) 0x3c00, (short) 0x3c00));
        assertEquals((short) 0x7c00,
                FloatConversions.mulIeeeFloat16((short) 0x7bff, (short) 0x4000));
        assertEquals((short) 0x0000,
                FloatConversions.mulIeeeFloat16((short) 0x0000, (short) 0x3c00));
        assertEquals((short) 0x7fff,
                FloatConversions.mulIeeeFloat16((short) 0x7c01, (short) 0x3c00));
    }
}
